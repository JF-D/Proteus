vocab_size=40478 #50257

mp_deg=1
pp_deg=1
cluster=clusters/dgx1_v100_1ib/n4_g8.json
bs=128

# NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH \
mpirun -np 1 python megatron_gpt.py -model gpt \
    -global-bs $bs -n-macro-batch 1 \
    -nlayer 12 -seq-length 1024 -hidden-size 1024 -nheads 16 -vocab-size $vocab_size \
    -ps dp -mp-deg $mp_deg -pp-deg $pp_deg -zero 0 --no-seq-first \
    -cluster $cluster \
    --profile-iters 50 #--bucket-size 19 #--checkpoint #--no-share-bandwidth #--no-seq-first #--reprofile
