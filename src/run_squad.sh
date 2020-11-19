python run_squad_warmup_proportion.py \
 --model_type bert \
 --model_name_or_path bert-base-uncased \
 --do_train \
 --do_eval \
 --version_2_with_negative \
 --data_dir ./squad2/ \
 --do_lower_case \
 --per_gpu_train_batch_size 8 \
 --per_gpu_eval_batch_size 8 \
 --learning_rate 3e-5 \
 --num_train_epochs 2.0 \
 --gradient_accumulation_steps 4\
 --max_seq_length 512 \
 --doc_stride 128 \
 --weight_decay 0.01 \
 --warmup_proportion 0.1 \
 --output_dir ./output/


 