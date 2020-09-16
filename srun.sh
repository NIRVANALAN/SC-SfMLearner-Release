#!/bin/bash
srun -p NTU --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=sfmlearner --kill-on-bad-exit=1 -w SG-IDC1-10-51-1-45  \
sh scripts/train_resnet18_depth_256.sh
