# Proteus runtime: ~210sec ; Actual runtime: ~22sec
python megatron_gpt.py -n-gpu-per-node 32 -global-bs 512 -n-macro-batch 4 -model gpt -nlayer 32 -seq-length 1024 -hidden-size 2560 -nheads 32 -vocab-size 50257 -make-vocab-size-divisible-by 128 -cluster clusters/dgx1_v100_1ib/cluster_info.json -ps pp -zero 1 -mp-deg 2 -pp-deg 1 -dom DM -bucket-size 100000 -profile-iters 10 -output_dir megatron_gpt_output

