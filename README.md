
# [EMNLP 2024] Breaking ReLU Barrier: Generalized Mixture of Experts for Dense Pretrained Models

This code is based on [MoEfication](https://github.com/thunlp/MoEfication).

We provide example script for MoEfying BLOOM. Code for MoEfying mBERT is also provided.

## Expert Construction

```bash
export model_path=BLOOM_PATH
export moe_workspace=WORKSPACE_PATH
export ffn_template='h.{}.mlp.dense_h_to_4h.weight'
export ffn_out_template='h.{}.mlp.dense_4h_to_h.weight'
export num_expert=64
export num_layers=24 # for bloom-1.7b
export last_layer=23 # for bloom-1.7b
python moefication/param_cluster_example.py --model_path=${model_path}/pytorch_model.bin --res_path=${moe_workspace}/ --templates=${ffn_template} --num-expert=${num_expert} --num-layer=${num_layers}

export corpus_file=CORPUS_PATH
export sample_size=10000
python examples/clm-inf-pattern-bloom.py --train_file=$corpus_file --model_name_or_path=${model_path} --res_path=${moe_workspace}/0 --templates=${ffn_template} --split_size=${sample_size} --start_split=0 --per_device_train_batch_size=1 # can be splitted into parallel runs by controlling split parameters
python examples/concat_multiple.py --res_path=${moe_workspace}/ --templates=${ffn_template} --num_layers=${num_layers}


python examples/calculate_mean.py --res_path=${moe_workspace}/ --templates=${ffn_out_template} --start_layer={layer_id} --end_layer={layer_id}
python moefication/mlp_select_example.py --config_path=${model_path} --mean_template=${ffn_out_template}_mean --templates=${ffn_template} --model_path=${model_path}/pytorch_model.bin --res_path=${moe_workspace}/ --num-expert=${num_expert} --start_layer=0 --end_layer=${last_layer}
```

Next, you may evaluate the model using the lm-evaluation-harness provided. The evaluation code is a modified version of [lm-evaluation-harness](https://github.com/OpenGPTX/lm-evaluation-harness).
