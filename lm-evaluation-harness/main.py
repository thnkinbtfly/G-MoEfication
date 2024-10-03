#!/usr/bin/env python
import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator, utils

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--no_tokenizer_check", action="store_true")
    parser.add_argument(
        "--write_detailed_eval_info", action="store_true", default=False
    )
    parser.add_argument("--detailed_eval_info_path", type=str, default=None)
    parser.add_argument("--bootstrap_iters", type=int, default=100000)
    parser.add_argument("--moe_res_path", type=str, default="")
    parser.add_argument("--moe_selection", type=int, default=48)
    parser.add_argument("--no_moe_use_mean", action="store_true", default=False)
    parser.add_argument("--moe_use_random_expert", action="store_true", default=False)
    parser.add_argument("--train_mean", action="store_true", default=False)
    parser.add_argument("--moe_use_mean_emb", action="store_true", default=False)
    parser.add_argument("--moe_experts", type=int, default=64)
    parser.add_argument("--no_moe_use_kmeans", action="store_true", default=False)
    parser.add_argument("--moe_mask_path", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        no_tokenizer_check=args.no_tokenizer_check,
        write_out=args.write_detailed_eval_info,
        output_base_path=args.detailed_eval_info_path,
        bootstrap_iters=args.bootstrap_iters,
        args=args,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
