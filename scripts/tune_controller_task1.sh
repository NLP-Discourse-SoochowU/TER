#!/bin/sh
# sed -i 's/\r$//' tune_controller_task1.sh
cd ../code
cuda_id=0

task1_pre_m=../exp/Controller_task1/z7SOg44r/model_task1.pth
for i in {1..5}
do
    # 1. Controller tuning
    data_dir=../data/entailment_trees_binary
    output_dir=../exp/Controller_task1/retrieve_learning_v5/round_$i
    mkdir output_dir
    CUDA_VISIBLE_DEVICES=$cuda_id python tune_Controller.py --task_name=task1 \
      --retrieve_data_file=$data_dir/task_1/control_state_seq.pkl \
      --train_data_file=$data_dir/train.controller.task1.v36.jsonl \
      --dev_data_file=$data_dir/dev.controller.task1.v36.jsonl \
      --model_name_or_path=albert-xxlarge-v2 --bs=1 --lr=1e-5 --epochs=10 --adafactor \
      --eval_epoch=2 --report_epoch=1 --code_dir=../code --exp_dir=$output_dir \
      --pre_m=$task1_pre_m --save_model --seed 2171

    # 2. Controller tuning
    data_dir=../data/Controller_data
    pretrained_ctl_dir=$output_dir
    pretrained_ctl_name=model_last.pth
    CUDA_VISIBLE_DEVICES=$cuda_id python train_Controller.py \
    --task_name task1 \
    --train_data_file $data_dir/train.controller.task1.v36.jsonl \
    --dev_data_file $data_dir/dev.controller.task1.v36.jsonl \
    --model_name_or_path albert-xxlarge-v2 \
    --bs 6 --lr 1e-5 --epochs 10 --adafactor \
    --eval_epoch 2 --report_epoch 1 \
    --code_dir ../code \
    --exp_dir $output_dir \
    --save_model --seed 2171 --consider_rst 1000 \
    --retrieve_learning --load_pre_trained_model --exp_dir_controller $pretrained_ctl_dir --model_name_controller $pretrained_ctl_name

    task1_pre_m=$output_dir/model_last.pth
done