import multiprocessing
from tempfile import template
import utils
import numpy as np
from transformers import AutoConfig, BloomConfig
from transformers.models.bloom.modeling_bloom import BloomGelu
import torch
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='results/t5-base/ckpt.bin', help='path to the model checkpoint')
parser.add_argument('--res_path', type=str, default='results/t5-base/', help='path to store the results of moefication')
parser.add_argument('--start_layer', type=int, default=0, help='number of layers')
parser.add_argument('--pool_size', type=int, default=1, help='number of layers')
parser.add_argument('--end_layer', type=int, default=11, help='number of layers')
parser.add_argument('--num-expert', type=int, default=96, help='number of experts')
parser.add_argument('--templates', type=str, default='encoder.block.{}.layer.1.DenseReluDense.wi.weight,decoder.block.{}.layer.2.DenseReluDense.wi.weight', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')
parser.add_argument('--use_abs', default=False, action='store_true', help='number of experts')
parser.add_argument('--force_use_square', default=False, action='store_true', help='number of experts')
parser.add_argument('--config_path', type=str, default='', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')
parser.add_argument('--mean_template', type=str, default='', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')

args = parser.parse_args()

config = utils.ModelConfig(args.model_path, args.res_path, split_num=args.num_expert)

config.mean_template = args.mean_template
config.use_abs = args.use_abs

model_config = AutoConfig.from_pretrained(args.config_path)
config.act_fn = None
config.force_use_square = args.force_use_square
for key in ['feed_forward_proj', 'activation_function', 'hidden_act']:
    if hasattr(model_config, key):
        config.act_fn = getattr(model_config, key)
        break
if isinstance(model_config, BloomConfig):
    config.act_fn = BloomGelu()
if config.act_fn is None:
    raise ValueError('Cannot find activation function in config')

templates = args.templates.split(',')
split_folder = 'param_split'

def run(i):
    center = utils.MLPCenter(config, template, '{}/{}/{}'.format(args.res_path, split_folder, template.format(i)), i)
    center.cal_center()

for template in templates:
    pool = multiprocessing.Pool(args.pool_size)
    pool.map(run, range(args.start_layer, args.end_layer + 1))
