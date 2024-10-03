import collections
import itertools
import numpy as np
import random
import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.utils import positional_deprecated, run_task_tests


import torch, os, types, pickle
import torch.nn.functional as F
import numpy as np

import torch

class RandomTensor(torch.nn.Module):
    def __init__(self, size, random_tensor=None):
        super().__init__()
        self.size = size
        tensor = random_tensor
        if tensor is None:
            tensor = torch.rand(size)
        else:
            print("LOADING RAND TENSOR FROM STATE DICT")
            print(tensor)
        self.register_buffer('random_tensor', tensor)

    def forward(self, input):
        return torch.broadcast_to(self.random_tensor, input.shape[:-1] + (self.size,))

class MaskTensor(torch.nn.Module):
    def __init__(self, mask_tensor):
        super().__init__()
        tensor = mask_tensor
        self.register_buffer('mask_tensor', tensor)

    def forward(self, input):
        return self.mask_tensor

def change_forward(model, res_path, num_experts, num_selection, use_mean=True, mean_ckpt_path="", train_mean=False, use_mean_emb=False,
                   use_kmeans=True, use_random_expert=False, mask_path=''):
    assert not (use_mean and use_mean_emb)
    def _forward(ffn_self, input):
        bsz, seq_len, hidden_size = input.shape
        hidden_states_mlp = input.clone().detach()
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size)

        hidden_states_mlp = hidden_states_mlp / torch.norm(hidden_states_mlp, dim=-1).unsqueeze(-1)
        score = ffn_self.mlp(hidden_states_mlp)

        if mask_path:
            labels = torch.topk(score, k=num_selection, dim=-1).indices
            cur_mask = torch.zeros_like(score, dtype=torch.bool)
            cur_mask[labels] = True
            cur_mask = torch.broadcast_to(cur_mask, (bsz, seq_len, cur_mask.shape[-1]))
            ffn_self.placeholder["cur_mask"] = cur_mask

        else:
            labels = torch.topk(score, k=num_selection, dim=-1)[1].view(bsz, seq_len, num_selection)
            unselected = torch.topk(-score, k=num_experts - num_selection, dim=-1)[1].view(bsz, seq_len, num_experts - num_selection)
            cur_mask = torch.nn.functional.embedding(labels, ffn_self.patterns).sum(-2)
            ffn_self.placeholder['unselected'] = unselected
            ffn_self.placeholder["cur_mask"] = cur_mask

        # ffn_self.res.append(input.detach().cpu())
        hidden_states = F.linear(input, ffn_self.weight, ffn_self.bias)
        hidden_states[cur_mask == False] = 0
        return hidden_states

    def _forward_mean(ffn_self, input):
        cur_mask = ffn_self.placeholder["cur_mask"]
        zeros = torch.zeros_like(input)
        input += torch.where(cur_mask == False, ffn_self.global_mean, zeros) # We need to add global_mean where curs_mask==0.
        hidden_states = F.linear(input, ffn_self.weight, ffn_self.bias)
        return hidden_states

    def _forward_mean_emb(ffn_self, input):
        added_bias = torch.nn.functional.embedding(ffn_self.placeholder['unselected'], ffn_self.mean_emb).sum(-2)
        hidden_states = F.linear(input, ffn_self.weight, ffn_self.bias)
        return hidden_states + added_bias

    layers = model.base_model.h
    state_dict = None
    if mask_path:
        masks = pickle.load(open(mask_path, 'rb'))
        for k in list(masks.keys()):
            masks[k] = masks[k].cpu()
    if mean_ckpt_path:
        state_dict = torch.load(mean_ckpt_path)
    for layer_idx, layer in enumerate(layers):
        ffn = layer.mlp.dense_h_to_4h
        placeholder = {}
        ffn.placeholder = placeholder
        name = ('h.{}.mlp.dense_h_to_4h.weight').format(layer_idx)
        path = os.path.join(res_path, 'param_split' if use_kmeans else 'gp_split', name)
        labels = torch.load(path)
        cluster_num = max(labels) + 1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        patterns = torch.Tensor(patterns).cuda()

        ffn.patterns = patterns
        ffn.k = num_selection
        if mask_path:
            print("LOADING MASK FROM PATH")
            mask_tensor = masks[f"h.{layer_idx}.mlp.dense_h_to_4h.weight"]
            ffn.mlp = MaskTensor(mask_tensor).cuda()
        elif not use_random_expert:
            ffn.mlp = torch.load(path + '_input_compl').cuda()
        else:
            random_tensor = None
            if state_dict is not None:
                random_tensor = state_dict[f"h.{layer_idx}.mlp.dense_h_to_4h.weight_random_tensor"]
            ffn.mlp = RandomTensor(size=num_experts, random_tensor=random_tensor).cuda()
        for p in ffn.mlp.parameters():
            p.requires_grad = False
        ffn.forward = types.MethodType(_forward, ffn)

        if use_mean:
            fn_name = 'register_parameter' if train_mean else 'register_buffer'

            ffn = layer.mlp.dense_4h_to_h
            ffn.placeholder = placeholder
            name = 'h.{}.mlp.dense_4h_to_h'.format(layer_idx)
            if mean_ckpt_path:
                print("Loading mean from ckpt")
                getattr(ffn, fn_name)("global_mean", torch.nn.Parameter(state_dict[name + '.global_mean'].float().cuda(), requires_grad=train_mean))
            else:
                print("Loading mean from file")
                path = os.path.join(res_path, name)
                print(path)
                getattr(ffn, fn_name)("global_mean", torch.nn.Parameter(torch.load(path + '.weight_mean').float().cuda(), requires_grad=train_mean))

            ffn.forward = types.MethodType(_forward_mean, ffn)

        elif use_mean_emb:
            fn_name = 'register_parameter' if train_mean else 'register_buffer'
            # tensor_fn = torch.nn.Parameter if train_mean else torch.Tensor
            tensor_fn = torch.nn.Parameter

            ffn = layer.mlp.dense_4h_to_h
            ffn.placeholder = placeholder
            name = 'h.{}.mlp.dense_4h_to_h'.format(layer_idx)
            if mean_ckpt_path:
                print("Loading mean emb from ckpt")
                getattr(ffn, fn_name)("mean_emb", tensor_fn(state_dict[name + '.mean_emb'], requires_grad=train_mean))
                print(ffn.mean_emb.requires_grad)
            else:
                print("Loading mean emb from file")
                path = os.path.join(res_path, name)
                mean_input_tensor = torch.load(path + '.weight_mean').float().cuda()
                output_mean_embs = []
                for pattern in patterns:
                    output_mean_tensor = mean_input_tensor.clone().detach()
                    output_mean_tensor[pattern == False] = 0
                    output_mean_tensor = torch.matmul(ffn.weight, output_mean_tensor)
                    output_mean_embs.append(output_mean_tensor)
                output_mean_embs = torch.stack(output_mean_embs, dim=0)
                getattr(ffn, fn_name)("mean_emb", tensor_fn(output_mean_embs, requires_grad=train_mean))
                # ffn.mean_emb.requires_grad = train_mean
                print(ffn.mean_emb.requires_grad)

            ffn.forward = types.MethodType(_forward_mean_emb, ffn)

@positional_deprecated
def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=0,
    batch_size=None,
    device=None,
    no_cache=False,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    check_integrity=False,
    decontamination_ngrams_path=None,
    no_tokenizer_check=False,
    write_out=False,
    output_base_path=None,
        args=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write details about prompts and logits to json for all tasks
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir.
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

    # this is just a temporary fix
    if isinstance(model, str):
        if model_args is None:
            model_args = ""

        additional_args = {
            "batch_size": batch_size,
            "device": device,
        }

        if model in ("hf", "gpt2"):
            additional_args["no_tokenizer_check"] = no_tokenizer_check

        lm = lm_eval.models.get_model(model).create_from_arg_string(
            model_args,
            additional_args,
        )

        if args.moe_res_path:
            change_forward(lm.model, args.moe_res_path, args.moe_experts, args.moe_selection, not args.no_moe_use_mean, '',
                           args.train_mean,
                           args.moe_use_mean_emb,
                           not args.no_moe_use_kmeans, args.moe_use_random_expert, args.moe_mask_path)

    else:
        assert isinstance(model, lm_eval.base.LM)
        lm = model

    if not no_cache:
        if args.moe_res_path:
            additional_hash = vars(args)
            import json
            additional_hash = json.dumps(additional_hash, sort_keys=True)
            # additional_hash = args.moe_res_path + args.moe_experts + args.moe_selection + str(
            #     not args.no_moe_use_mean) + str(args.train_mean) + str(args.moe_use_mean_emb) + str(
            #     not args.no_moe_use_kmeans) + str(args.moe_use_random_expert) + str(args.moe_mask_path)
            from hashlib import blake2b
            additional_hash = blake2b(additional_hash.encode('utf-8'), digest_size=24).hexdigest()
            print("HASH: ", additional_hash)
        else:
            additional_hash = ""

        lm = lm_eval.base.CachingLM(
            lm,
            "lm_cache/"
            + model
            + "_" + ','.join(tasks) + "_" + str(num_fewshot) + "_"
            + model_args.replace("=", "-").replace(",", "_").replace("/", "-") + additional_hash
            + ".db",
        )

    task_dict = lm_eval.tasks.get_task_dict(tasks)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        output_base_path=output_base_path,
    )

    # add info about the model and few shot config
    results["config"] = {
        "model": model,
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "device": device,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
        "description_dict": description_dict,
    }

    return results


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    provide_description=None,
    num_fewshot=0,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param write_out: bool
        If True, write all prompts, logits and metrics to json for offline analysis
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir
    :return
        Dictionary of results
    """
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

    # TODO: todo: implement proper description-providing system
    assert not provide_description  # not implemented.
    if provide_description is not None:
        # nudge people to not specify it at all
        print(
            "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
        )

    decontaminate = decontamination_ngrams_path is not None

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}

    # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
    # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
    # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
    #  - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}
    write_out_info = {}

    docs_for_decontamination = collections.defaultdict(list)

    # get lists of each type of request
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)
        print(f"Task: {task_name}; number of docs: {len(task_docs)}")

        if write_out:
            prompt_details = []

        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )
        if limit is not None:
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
            if decontaminate and task.should_decontaminate():
                docs_for_decontamination[(task_name, task_set)].append(
                    task.doc_to_decontamination_query(doc)
                )

            docs[(task_name, doc_id)] = doc
            ctx = task.fewshot_context(
                doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
            )
            reqs = task.construct_requests(doc, ctx)

            if write_out:
                prompt_details.append({"doc_id": doc_id})

            # print the prompt for the first few documents
            if doc_id < 1:
                print(
                    f"Task: {task_name}; document {doc_id}; context prompt (starting on next line):\n{ctx}\n(end of prompt on previous line)"
                )
                print("Requests:", reqs)

            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append((i, task_name, doc, doc_id))

                if write_out:
                    prompt_details[-1][f"prompt_{i}"] = "".join(
                        (map(lambda x: "".join(x), req.args))
                    )

        if write_out:
            write_out_info[task_name] = prompt_details

    # Compare all tasks/sets at once to ensure a single training set scan
    if decontaminate:
        from lm_eval.decontamination.decontaminate import get_train_overlap

        print("Finding train/test overlap, please wait...")
        overlaps = get_train_overlap(
            docs_for_decontamination, decontamination_ngrams_path, limit
        )

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)

    # execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
        #       only in index. We could implement some kind of caching, but that would be more of a band-aid
        #       solution. we could also implement some kind of auto-grouping here;
        #       they should end up next to each other.

        print("Running", reqtype, "requests")
        resps = getattr(lm, reqtype)([req.args for req in reqs])
        resps = [
            x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
        ]

        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))

            if write_out:
                write_out_info[task_name][doc_id][f"logit_{i}"] = resp
                task = task_dict[task_name]
                if isinstance(task, lm_eval.base.MultipleChoiceTask):
                    write_out_info[task_name][doc_id]["truth"] = doc["gold"]
                elif isinstance(task, lm_eval.tasks.winogrande.Winogrande):
                    write_out_info[task_name][doc_id]["truth"] = task.answer_to_num[
                        doc["answer"]
                    ]
                elif isinstance(task, lm_eval.tasks.opengptx.wino_x.WinograndeXDe):
                    write_out_info[task_name][doc_id]["truth"] = task.answer_to_num[
                        doc["answer"]
                    ]
                else:
                    write_out_info[task_name][doc_id]["truth"] = task.doc_to_target(doc)

    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for (task_name, doc_id), requests in process_res_queue.items():
        requests.sort(key=lambda x: x[0])
        requests = [x[1] for x in requests]

        task = task_dict[task_name]
        doc = docs[(task_name, doc_id)]

        metrics = task.process_results(doc, requests)
        for metric, value in metrics.items():
            vals[(task_name, metric)].append(value)

            if write_out:
                write_out_info[task_name][doc_id][metric] = str(value)

            # Re-use the evaluation for the decontaminated set by just ignoring the overlaps
            if decontaminate and task_name in overlaps:
                if doc_id not in overlaps[task_name]:
                    vals[(task_name, metric + decontaminate_suffix)].append(value)

    # aggregate results
    for (task_name, metric), items in vals.items():
        task = task_dict[task_name]
        real_metric = metric  # key when looking up the metric with task.aggregation
        if metric.endswith(decontaminate_suffix):
            real_metric = metric.replace(
                decontaminate_suffix, ""
            )  # decontaminated still uses the same metric
        results[task_name][metric] = task.aggregation()[real_metric](items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this

        stderr = lm_eval.metrics.stderr_for_metric(
            metric=task.aggregation()[real_metric],
            bootstrap_iters=min(bootstrap_iters, 1000)
            if metric in ["bleu", "chrf", "ter"]
            else bootstrap_iters,
        )

        if stderr is not None:
            results[task_name][metric + "_stderr"] = stderr(items)

    if write_out:
        import json
        import datetime
        import pathlib

        output_base_path = (
            pathlib.Path(output_base_path)
            if output_base_path is not None
            else pathlib.Path(".")
        )
        try:
            output_base_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass

        timestamp = datetime.datetime.utcnow().strftime("%d%m%Y-%H-%M-%S")

        for task_name, _ in task_dict_items:
            with open(
                output_base_path.joinpath(
                    f"{task_name}_detailed_eval_info_{timestamp}.json"
                ),
                "w",
                encoding="utf8",
            ) as fp:
                json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)

    return {"results": dict(results), "versions": dict(versions)}


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()
