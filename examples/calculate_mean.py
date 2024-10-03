import collections
from tqdm.auto import tqdm
import math
import os, glob
import types
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, MBartTokenizer, MBartTokenizerFast, default_data_collator, DataCollatorForSeq2Seq, \
    MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer, AutoModelForPreTraining

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--res_path', type=str, default='mbert_encoder_output', help='path to store the results of moefication')
parser.add_argument('--end_layer', type=int, default=11, help='path to store the results of moefication')
parser.add_argument('--start_layer', type=int, default=0, help='path to store the results of moefication')
parser.add_argument('--templates', type=str, default='encoder.block.{}.layer.1.DenseReluDense.wi.weight,decoder.block.{}.layer.2.DenseReluDense.wi.weight', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')
args = parser.parse_args()

for layer in range(args.start_layer, args.end_layer + 1):
    tensor_list = []
    layer_name = args.templates.format(layer)
    for f in tqdm(sorted(glob.glob(os.path.join(args.res_path, "*", layer_name)))[:30]):
        tensor = torch.load(f)
        if isinstance(tensor, torch.Tensor):
            tensor_list.append(tensor)
    tensor = torch.stack(tensor_list)
    torch.save(torch.mean(tensor, dim=(0,1)), f"{args.res_path}/{layer_name}_mean")
    del tensor
