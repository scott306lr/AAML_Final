# CFU Playground - AAML Final Project - Group 4

## Setup Project

About how to setup the project, please reference to [README.md in the root directory](https://github.com/scott306lr/AAML_Final/blob/main/README.md).

## Training Code

The final compressed model `final_0.875_qat_model.tflite` is located in the `model_compression` directory.
If you would like to reproduce the results, please follow the steps below:

Install the required packages:

```bash

# library for pruning
pip install torch-pruning 
# library for quantization + convert to tflite
pip install git+https://github.com/alibaba/TinyNeuralNetwork.git
```

There are three steps to build the final compressed model:

1. Train the baseline model
2. Prune and finetune the baseline model
3. Apply Quantization-Aware Training(QAT) then convert the model to tflite.

```bash
# Enter the project directory
cd model_compression

# Train the baseline model
torchrun --nproc_per_node=2 training/train.py --model resnet8 --data-path ./cifar10 --opt adamw --batch-size 128 --lr 1e-2 --lr-scheduler cosineannealinglr --auto-augment ta_wide --lr-warmup-epochs 3 --lr-warmup-method linear --epochs 500 --weight-decay 1e-4 --norm-weight-decay 0.0 --label-smoothing 0.0 --mixup-alpha 0.0 --cutmix-alpha 0.0 --random-erase 0.0 --wandb --sync-bn

# Prune the baseline model
python pruning/main.py --mode prune --model resnet8 --batch-size 128 --dataset cifar10 --method group_sl --speed-up 1.2 --global-pruning --reg 1e-4 --total-epochs 100 --sl-total-epochs 5  --restore <path_to_trained_model> --wandb

# Prune again on the pruned model
python pruning/main.py --mode prune --model resnet8 --batch-size 128 --dataset cifar10 --method group_sl --speed-up 1.3 --global-pruning --reg 1e-4 --total-epochs 100 --sl-total-epochs 5 --restore <path_to_trained_model> --load-pruned <path_to_pruned_model> --wandb 

# Quantize and convert the model to tflite
python quick_start_for_expert.py --model-path <path_to_pruned_model>
```

## Reference
The training scripts are modified from the following repositories:
[Torchvision - Image classification reference training scripts](https://github.com/pytorch/vision/tree/main/references/classification)
[Torch-Pruning - benchmarks](https://github.com/VainF/Torch-Pruning/tree/master/benchmarks)
[TinyNeuralNetwork - examples](https://github.com/alibaba/TinyNeuralNetwork/tree/23b02a3f3fd57adaa303be4aaab313f9ab70f83e)

