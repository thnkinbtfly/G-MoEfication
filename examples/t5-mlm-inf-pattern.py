import math
import os
import types
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, MBartTokenizer, MBartTokenizerFast, default_data_collator, DataCollatorForSeq2Seq, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer, AutoModelForPreTraining
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='t5-base', help='model name in huggingface model hub')
parser.add_argument('--split_size', type=int, default=None, help='model name in huggingface model hub')
parser.add_argument('--res_path', type=str, default='results/t5-base/', help='path to store the results of moefication')
parser.add_argument('--num-layer', type=int, default=12, help='number of layers')
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
model = AutoModelForPreTraining.from_pretrained(args.model_name_or_path).cuda()


from tfrecord.torch.dataset import TFRecordDataset

tfrecord_path = args.train_file
index_path = None
name_to_features = {
    "input_ids":
        "int",
    "input_mask":
        "int",
    "segment_ids":
        "int",
    "masked_lm_positions":
        "int",
    "masked_lm_ids":
        "int",
    "masked_lm_weights":
        "float",
    "next_sentence_labels":
        "int",
    "language":
        "int",
}


def decode(features):
    # get BGR image from bytes
    features["attention_mask"] = torch.from_numpy(features.pop("input_mask"))
    features["token_type_ids"] = torch.from_numpy(features.pop("segment_ids"))
    labels = torch.full((features["input_ids"].shape[0],), -100, dtype=torch.long)
    labels[torch.from_numpy(features.pop("masked_lm_positions"))] = torch.from_numpy(features.pop("masked_lm_ids"))
    features["labels"] = labels
    features["next_sentence_label"] = torch.from_numpy(features.pop("next_sentence_labels"))
    features.pop("language")
    features.pop("masked_lm_weights")
    return features


train_dataset = TFRecordDataset(tfrecord_path, index_path, name_to_features, transform=decode)

# Data collator
# This one will take care of randomly masking the tokens.
data_collator = default_data_collator

# DataLoaders creation:
# assert args.use_default_collator, "Only default collator is supported!! The version before is corrupted.."
train_dataloader = DataLoader(
    train_dataset,
    # shuffle=True,
    collate_fn=data_collator, batch_size=args.per_device_train_batch_size
)
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
    for template in args.templates.split(','):
        if 'encoder' in template:
            layers = model.base_model.encoder.layer
        else:
            raise NotImplementedError
        for layer_idx, layer in enumerate(layers):
            ffn = layer.intermediate.dense
            name = 'bert.encoder.layer.{}.intermediate.dense.weight'.format(layer_idx)
            res[name] = []
            modify_ffn(ffn, res[name])

            ffn = layer.output.dense
            name = 'bert.encoder.layer.{}.output.dense.weight'.format(layer_idx)
            res[name] = []
            res[name+'_out'] = []
            modify_ffn(ffn, res[name], res[name+'_out'])

    return res
        
res = change_forward(model)

# sst2 evaluation
for idx, batch in enumerate(tqdm(train_dataloader)):
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