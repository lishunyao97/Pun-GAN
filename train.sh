#!/bin/bash
step=0
n=158 # number of sample files in /Pun_Generation/data/samples
lr=0.01 # learning rate
while(( $step<800 )) # n*5 for 5 epochs
do
    sample_num=$[$step%$n]
    echo $step
    echo '[step 1] backward step one'
    python -u ./Pun_Generation/code/nmt.py \
--infer_batch_size=8 \
--vocab_prefix=./Pun_Generation/data/1backward/vocab \
--src=in \
--tgt=out \
--out_dir=./Pun_Generation/code/backward_model_path \
--dev_prefix=./Pun_Generation/data/1backward/dev \
--test_prefix=./Pun_Generation/data/1backward/test \
--sample_prefix=./Pun_Generation/data/samples/sample_"$sample_num" \
--sampling_temperature=1.0 \
--beam_width=0 \
--sample_size=32 \
--first_step=1.0 \
--learning_rate="$lr" > ./Pun_Generation/code/output1.txt

    echo '[step 2] forward rl'
    python -u ./Pun_Generation_Forward/code/nmt.py \
--infer_batch_size=64 \
--vocab_prefix=./Pun_Generation_Forward/data/2forward/vocab \
--src=in \
--tgt=out \
--out_dir=./Pun_Generation_Forward/code/forward_model_path \
--train_prefix=./Pun_Generation_Forward/data/2forward/forward_index \
--dev_prefix=./Pun_Generation_Forward/data/2forward/dev \
--test_prefix=./Pun_Generation_Forward/data/2forward/test \
--sample_prefix=./Pun_Generation/data/1backward/backward_step1.out \
--reward_prefix=./Pun_Generation_Forward/data/2forward/wsd_train_reward.in \
--sampling_temperature=-1.0 \
--beam_width=0 \
--sample_size=1 \
--learning_rate="$lr" > ./Pun_Generation_Forward/code/output1_sp.txt

    echo '[step 3] backward step two'
    python -u ./Pun_Generation/code/nmt.py \
--infer_batch_size=64 \
--vocab_prefix=./Pun_Generation/data/1backward/vocab \
--src=in \
--tgt=out \
--out_dir=./Pun_Generation/code/backward_model_path \
--train_prefix=./Pun_Generation/data/1backward/backward_step2 \
--dev_prefix=./Pun_Generation/data/1backward/dev \
--test_prefix=./Pun_Generation/data/1backward/test \
--sample_prefix=./Pun_Generation/data/sample_2548 \
--reward_prefix=./Pun_Generation_Forward/data/2forward/wsd_train_reward.in \
--beam_width=10 \
--sample_size=1 \
--learning_rate="$lr" > ./Pun_Generation/code/output1.txt

    python -u ./Pun_Generation/code/nmt.py \
--infer_batch_size=64 \
--vocab_prefix=./Pun_Generation/data/1backward/vocab \
--src=in \
--tgt=out \
--out_dir=./Pun_Generation/code/backward_model_path \
--train_prefix=./Pun_Generation/data/1backward/train \
--dev_prefix=./Pun_Generation/data/1backward/dev \
--test_prefix=./Pun_Generation/data/1backward/test \
--inference_input_file=./Pun_Generation/data/sample_2548 \
--inference_output_file=./Pun_Generation/code/backward_model_path/first_part_file \
--beam_width=10 \
--num_translations_per_input=1 > ./Pun_Generation/code/output_infer.txt

    python ./Pun_Generation/code/dealt.py 2548

    python -u ./Pun_Generation_Forward/code/nmt.py \
--infer_batch_size=64 \
--vocab_prefix=./Pun_Generation_Forward/data/2forward/vocab \
--src=in \
--tgt=out \
--out_dir=./Pun_Generation_Forward/code/forward_model_path \
--train_prefix=./Pun_Generation_Forward/data/2forward/train \
--dev_prefix=./Pun_Generation_Forward/data/2forward/dev \
--test_prefix=./Pun_Generation_Forward/data/2forward/test \
--inference_input_file=./Pun_Generation/code/backward_model_path/dealt_first_part_file \
--inference_output_file=./Pun_Generation_Forward/code/forward_model_path/second_part_file \
--beam_width=10 \
--num_translations_per_input=1 > ./Pun_Generation_Forward/code/output_infer.txt

    python ./Pun_Generation/code/concatenate.py

    let "step++"
done
