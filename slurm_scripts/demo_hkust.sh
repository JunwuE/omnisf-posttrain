#!/bin/bash

#SBATCH -p acd
#SBATCH --job-name=speed_test
#SBATCH -o /data/user/user11/project/PRM2/slurm_log/%j.out # slurm的输出文件，%j是jobid
#SBATCH --nodes=2               # 请求2个节点
#SBATCH --ntasks-per-node=4     #每个节点上4个任务
#SBATCH --gres=gpu:4            # 每个节点4个GPU

module load anaconda3
module load cuda/12.4

export NCCL_SOCKET_IFNAME=vlan0.2135
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=138
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=4
export NCCL_NVLS_ENABLE=0
export NCCL_NVLS_PLUGIN=1
export NCCL_NVLS_LANES=2
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_MIN_CTAS=32
export NCCL_MAX_CTAS=128
export NCCL_IB_RETRY_CNT=7
export NCCL_MIN_NCHANNELS=64
export NCCL_MAX_NCHANNELS=256
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_BUFFSIZE=33554432
export NCCL_LL_BUFFSIZE=33554432
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_P2P_LEVEL=nvl
export NCCL_ALGO=nvlstree,ring
export NCCL_LL_MAX_NCHANNELS=4
export NCCL_CROSS_NIC=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_SHM_DISABLE=0
export NCCL_COLLNET_ENABLE=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export NCCL_IB_TIMEOUT=3600


# 以下是作业要执行的命令
echo "Job started at $(date)"
python your_script.py  # 假设要运行一个Python脚本
echo "Job ended at $(date)"