import os
import argparse
import subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-partition', type=str, default='pat_dev')
parser.add_argument('-model', type=str, default='alexnet')
parser.add_argument('-version', type=str, default=None)
parser.add_argument('-bs', type=int, default=256)
parser.add_argument('-ps', type=str, default='dp')
parser.add_argument('-bucket-size', type=int, default=25)
parser.add_argument('-spec-dev', type=str, default='')
parser.add_argument('-cluster', type=str, default='slurm') # slurm, mpirun
parser.add_argument('--parse-log', action='store_true')
args = parser.parse_args()


def median(lst):
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2

    if (lstLen % 2):
        return sortedLst[index]
    else:
        return (sortedLst[index] + sortedLst[index + 1]) / 2.0


if __name__ == '__main__':
    basecmd = f'python pytorch_profile.py -model {args.model} -bs {args.bs} -ps {args.ps} '
    basecmd += f'-bucket-size {args.bucket_size} '
    if args.cluster == 'slurm':
        basecmd += '-launch slurm'
    elif args.cluster == 'mpirun':
        basecmd += '-launch mpirun'
    if args.version is not None:
        basecmd += f' -version {args.version}'

    os.makedirs(f'log/{args.model}', exist_ok=True)
    nreplications = 3
    ndevs = [1, 2, 4, 8]
    # ndevs = [1, 2, 4, 8, 16, 32]

    speeds = []
    input_mems, model_mems, mem_loads, work_mems = [], [], [], []
    for ndev in ndevs:
        rep_speeds = []
        proc_input, proc_model, proc_mem, proc_work = [], [], [], []
        for r in range(nreplications):
            g = ndev if ndev < 8 else 8
            if args.cluster == 'slurm':
                prefix = f'srun -p {args.partition} -n {ndev} {args.spec_dev} --gres=gpu:{g} --ntasks-per-node={g} '
            elif args.cluster == 'mpirun':
                prefix = f'mpirun --bind-to none -np {ndev} '
            cmd = prefix + basecmd
            logfile = f'log/{args.model}/{args.model}_bs{args.bs}_{args.ps}_n{ndev}_r{r}.log'
            cmd += f' | tee {logfile}'
            if not args.parse_log:
                print(cmd)
                subprocess.run(cmd, shell=True)

            iter_speeds = []
            with open(logfile, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line.startswith('speed:'):
                        a = float(line.split()[1][:-7])
                        iter_speeds.append(a)
                    elif line.startswith('Input:'):
                        a = line.split()
                        input_mem = float(a[1][:-3])
                        model_mem = float(a[3][:-3])
                        mem_load = float(a[5][:-3])
                        work_mem = float(a[7][:-2])

                        proc_input.append(input_mem)
                        proc_model.append(model_mem)
                        proc_mem.append(mem_load)
                        proc_work.append(work_mem)
                    elif 'speed:' in line:
                        a = float(line.split()[2][:-7])
                        iter_speeds.append(a)
            if len(iter_speeds) != 0:
                rep_speeds.append(median(iter_speeds))
                # rep_speeds.append(np.min(iter_speeds))
        speeds.append(np.mean(rep_speeds))
        # speeds.append(min(rep_speeds))
        input_mems.append(max(proc_input))
        model_mems.append(max(proc_model))
        mem_loads.append(max(proc_mem))
        work_mems.append(max(proc_work))

    print(
        '=========================== Statistics ==============================='
    )
    print(f'model: {args.model}, bs: {args.bs}')
    print()
    print(
        f'ndev,  speed(ms/iter),  Input (MB),  Model (MB),  MemoryLoad (MB),  WorkMemory (MB)'
    )
    for i, ndev in enumerate(ndevs):
        speed = speeds[i]
        input_mem = input_mems[i]
        model_mem = model_mems[i]
        mem_load = mem_loads[i]
        work_mem = work_mems[i]
        print(
            f'{ndev:^4},  {speed:11.4f},  {input_mem:11.4f},  {model_mem:10.4f},  '
            f'{mem_load:12.4f},  {work_mem:16.4f}')
