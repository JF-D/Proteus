python alexnet.py -model alexnet -bs 256 -cluster clusters/dgx1_v100_2ib/n1_g1.json -ps dp --profile-iters 50 --reprofile
sleep 10s
python alexnet.py -model resnet50 -bs 32 -cluster clusters/dgx1_v100_2ib/n1_g1.json -ps dp --profile-iters 50 --reprofile
sleep 10s
python alexnet.py -model inception_v3 -bs 32 -cluster clusters/dgx1_v100_2ib/n1_g1.json -ps dp --profile-iters 50 --reprofile
sleep 10s
python megatron_gpt.py -bs 4 -version layer -cluster clusters/dgx1_v100_2ib/n1_g1.json --bucket-size 35 --profile-iters 50 --reprofile
