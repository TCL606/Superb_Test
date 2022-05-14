#!/bin/bash

stage=1

goahead=0

s3prl_path=/mnt/lustre/sjtu/home/xc915/superb/s3prl/s3prl
expdir=/data/tangchangli/superb/mixed_data2vec_hubert_final/IC
ExpName=IC
upstream=mixed_data2vec_hubert
downstream=fluent_commands
config=/mnt/lustre/sjtu/home/xc915/superb/s3prl/s3prl/downstream/ctc/snips.yaml
overwrite="config.downstream_expert.datarc.batch_size=16,,config.downstream_expert.datarc.train_batch_size=16"
gpus=1
distributed="-m torch.distributed.launch --nproc_per_node ${gpus}"
test_ckpt=dev-best.ckpt

if [ $stage -eq -1 ]
then
    echo "Stage -1: Kill failed DDP task"

    pkill -f "mnt/lustre/sjtu/home/xc915/bin/miniconda/envs/superb/bin/python3 -u run_downstream.py --local_rank=1 -m train -n Debug -u data2vec_ffn -d asr -p /data/tangchangli/superb/data2vec_ffn_base/ASR -o config.downstream_expert.datarc.batch_size=16,,config.downstream_expert.datarc.train_batch_size=16"
    #-c $config \
fi

if [ $stage -eq 0 ]
then
    echo "Stage 0: Resume training"

    cd $s3prl_path && python3 run_downstream.py \
    -m train \
    -e $expdir/$test_ckpt 
    #-e $ExpName/$test_ckpt
fi

if [ $stage -eq 1 ]
then
    echo "Stage 1: finetune on superb"

    if [ $gpus -eq 1 ]
    then
        cd $s3prl_path && python3 run_downstream.py \
        -m train \
        -n $ExpName \
        -u $upstream \
        -d $downstream \
        -p $expdir 
        #-c $config 
        #-o $overwrite \
    else
        cd $s3prl_path && python3 $distributed run_downstream.py \
        -m train \
        -n $ExpName \
        -u $upstream \
        -d $downstream \
        -p $expdir \
        -o $overwrite 
        #-c $config \
    fi
    if [ $goahead -eq 1 ]
    then
        stage=$((stage+1))
    fi
fi

if [ $stage -eq 2 ]
then
    echo "Stage 2: Test on superb"

    cd $s3prl_path && python3 run_downstream.py \
    -m evaluate \
    -e $expdir/$test_ckpt 
    #-e $ExpName/$test_ckpt
fi

echo "finished"