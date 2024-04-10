import os
import argparse
import subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-partition', type=str, default='pat_dev')
parser.add_argument('-model', type=str, default='alexnet')
parser.add_argument('-version', type=str, default=None)
parser.add_argument('-bs', type=int, default=256)
parser.add_argument('-gpu', type=str, default='1080')
args = parser.parse_args()

if __name__ == '__main__':
    basecmd = f'python pytorch_profile.py -model {args.model} -bs {args.bs} --slurm'
    if args.version is not None:
        basecmd += f' -version {args.version}'

    os.makedirs(f'log/timeline', exist_ok=True)
    ndevs = [1, 2, 4, 8, 16, 32, 64]

    for ndev in ndevs:
        g = ndev if ndev < 8 else 8
        srun_prefix = f'srun -p {args.partition} -n {ndev} --gres=gpu:{g} --ntasks-per-node={g} '
        cmd = srun_prefix + basecmd + f' --timeline --timeline-name log/timeline/{args.model}_bs{args.bs}_{args.gpu}_prof_n{ndev}'
        subprocess.run(cmd, shell=True)
