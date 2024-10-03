import glob, os
from tqdm.auto import tqdm

import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--res_path', type=str, default='results/t5-base/', help='path to store the results of moefication')
parser.add_argument('--num_layers', type=int, default=12, help='path to store the results of moefication')
parser.add_argument('--templates', type=str, default='encoder.block.{}.layer.1.DenseReluDense.wi.weight,decoder.block.{}.layer.2.DenseReluDense.wi.weight', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')
args = parser.parse_args()

for layer in range(args.num_layers):
    for layer_template in args.templates.split(','):
        layer_name = layer_template.format(layer)
        print(layer_name)
        tensors = []
        for f in tqdm(glob.glob(os.path.join(args.res_path, "*", layer_name))):
            tensor = torch.load(f)
            if isinstance(tensor, torch.Tensor):
                tensors.append(torch.load(f))

        print(len(tensors))
        tensor = torch.cat(tensors, dim=0)
        torch.save(tensor, os.path.join(args.res_path, layer_name))

