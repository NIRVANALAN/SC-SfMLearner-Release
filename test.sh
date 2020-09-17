#!/bin/bash
srun -p NTU --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=test_sfm --kill-on-bad-exit=1 -w SG-IDC1-10-51-1-45  \
sh scripts/test_kitti_depth.sh