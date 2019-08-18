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
