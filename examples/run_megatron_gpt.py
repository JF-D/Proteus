import os
import sys
import json
import argparse

def construct_command_line_args_str(config):
    # all the null values to None
    # for k, v in config.items():
    #     if v == 'null':
    #         config[k] = None
    # return ' '.join([f'-{k} {v}' for k, v in config.items()])

    # if value is null, then don't include it in the command line args
    return ' '.join([f'-{k} {v}' for k, v in config.items() if (v != 'null' and v != None)])


if __name__ == '__main__':
    # --cluster "$1" --config "$2" --output_dir "$3"
    print("Hello from run_megatron_gpt.py", flush = True)

    parser = argparse.ArgumentParser()
    # parser.add_argument('-cluster', type=str, default='clusters/dgx1_v100_1ib/cluster_info.json')
    parser.add_argument('--config', type=str,  default='config.json')
    parser.add_argument('--output_dir', type=str , default='megatron_gpt_output')
    args = parser.parse_args()

    config = json.load(open(args.config))
    cluster = config['cluster']

    print(f"Content of config: {json.load(open(args.config))}")

    cmd_line_args = construct_command_line_args_str(json.load(open(args.config)))

    cmd = f'python megatron_gpt.py {cmd_line_args} -output_dir {args.output_dir}'

    print(f'Running command: {cmd}')
    os.system(cmd)