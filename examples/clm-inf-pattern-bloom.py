import math
import os
import types
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, MBartTokenizer, MBartTokenizerFast, default_data_collator, DataCollatorForSeq2Seq, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer, AutoModelForPreTraining, LlamaTokenizer, LlamaForCausalLM, LlamaConfig, AutoModelForCausalLM
from itertools import chain
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from transformers import set_seed
set_seed(42)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='t5-base', help='model name in huggingface model hub')
parser.add_argument('--split_size', type=int, default=None, help='model name in huggingface model hub')
parser.add_argument('--start_split', type=int, default=None, help='model name in huggingface model hub')
parser.add_argument('--res_path', type=str, default='results/t5-base/', help='path to store the results of moefication')
parser.add_argument('--num-layer', type=int, default=12, help='number of layers')
parser.add_argument('--block_size', type=int, default=512, help='number of layers')
parser.add_argument('--templates', type=str, default='encoder.block.{}.layer.1.DenseReluDense.wi.weight,decoder.block.{}.layer.2.DenseReluDense.wi.weight', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')
parser.add_argument(
    "--per_device_train_batch_size",
    type=int,
    default=1,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--preprocessing_num_workers",
    type=int,
    default=None,
    help="The number of processes to use for the preprocessing.",
)
parser.add_argument(
    "--ignore_pad_token_for_loss",
    type=bool,
    default=True,
    help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
)
parser.add_argument("--train_file", type=str, default=None, help="Source language id for translation.")
args = parser.parse_args()
if not os.path.exists(args.res_path):
    os.makedirs(args.res_path)

from accelerate import Accelerator
accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
config = AutoConfig.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_safetensors=False).cuda()

data_files = {}
dataset_args = {}
if args.train_file is not None:
    data_files["train"] = args.train_file
extension = args.train_file.split(".")[-1]
if extension == "txt":
    extension = "text"
raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
# If no validation data is there, validation_split_percentage will be used to divide the dataset.

# Preprocessing the datasets.
# First we tokenize all the texts.
column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]


def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

with accelerator.main_process_first():
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

if args.block_size is None:
    block_size = tokenizer.model_max_length
    if block_size > 1024:
        print(
            "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
            " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
            " override this default with `--block_size xxx`."
        )
    block_size = 1024
else:
    if args.block_size > tokenizer.model_max_length:
        print(
            f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(args.block_size, tokenizer.model_max_length)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

with accelerator.main_process_first():
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        desc=f"Grouping texts in chunks of {block_size}",
    )

train_dataset = lm_datasets["train"]
# eval_dataset = lm_datasets["validation"]

# DataLoaders creation:
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
)
# eval_dataloader = DataLoader(
#     eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
# )

# eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

train_dataloader, model = accelerator.prepare(train_dataloader, model)
# eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


# pred = []

def change_forward(model):

    def _forward(ffn_self, input):
        ffn_self.res.append(input.detach().cpu())
        output = F.linear(input, ffn_self.weight, ffn_self.bias)
        if ffn_self.res_out is not None:
            ffn_self.res_out.append(output.detach().cpu())
        return output

    def modify_ffn(ffn, res, res_out=None):
        # assert type(ffn) == T5DenseActDense
        ffn.res = res
        ffn.res_out = res_out
        ffn.forward = types.MethodType(_forward, ffn)   

    # encoder
    res = {}
    layers = model.base_model.h
    for layer_idx, layer in enumerate(layers):
        ffn = layer.mlp.dense_h_to_4h
        name = 'h.{}.mlp.dense_h_to_4h.weight'.format(layer_idx)
        res[name] = []
        modify_ffn(ffn, res[name])

        ffn = layer.mlp.dense_4h_to_h
        name = 'h.{}.mlp.dense_4h_to_h.weight'.format(layer_idx)
        res[name] = []
        modify_ffn(ffn, res[name])

    return res
        
res = change_forward(model)

# sst2 evaluation
with torch.no_grad():
    for idx, batch in enumerate(tqdm(train_dataloader)):
        if idx <= math.ceil(args.start_split / args.per_device_train_batch_size):
            continue
        if idx == math.ceil(args.split_size / args.per_device_train_batch_size):
            break

        output = model(**batch)

#     pred.append(int(output.logits[:, 0, 1465].item() > output.logits[:, 0, 2841].item()) == instance['label'])
#
# print("Acc", sum(pred) * 1. / len(pred))

for k, v in res.items():
    v = [x.reshape(-1, x.shape[-1]) for x in v]
    v = torch.cat(v, dim=0)
    torch.save(v, os.path.join(args.res_path,k))