#!/bin/sh
# sed -i 's/\r$//' tune_controller_task2.sh
cd ../code
cuda_id=2

task2_pre_m=../exp/Controller_task2/NuCkQlfx/model_task2.pth
for i in {1..5}
do
    # 1. ours retrieving
    data_dir=../data/entailment_trees_binary
    output_dir=../exp/Controller_task2/retrieve_learning_v5/round_$i
    mkdir output_dir
    CUDA_VISIBLE_DEVICES=$cuda_id python tune_Controller.py --task_name=task2 \
      --retrieve_data_file=$data_dir/task_2/control_state_seq.pkl \
      --train_data_file=$data_dir/train.controller.task2.v36.jsonl \
      --dev_data_file=$data_dir/dev.controller.task2.v36.jsonl \
      --model_name_or_path=albert-xxlarge-v2 --bs=1 --lr=1e-5 --epochs=10 --adafactor \
      --eval_epoch=2 --report_epoch=1 --code_dir=../code --exp_dir=$output_dir \
      --pre_m=$task2_pre_m --save_model --seed 2171

    # 2. original + rst
    data_dir=../data/Controller_data
    pretrained_ctl_dir=$output_dir
    pretrained_ctl_name=model_last.pth
    CUDA_VISIBLE_DEVICES=$cuda_id python train_Controller.py \
    --task_name task2 \
    --train_data_file $data_dir/train.controller.task2.v36.jsonl \
    --dev_data_file $data_dir/dev.controller.task2.v36.jsonl \
    --model_name_or_path albert-xxlarge-v2 \
    --bs 6 --lr 1e-5 --epochs 10 --adafactor \
    --eval_epoch 2 --report_epoch 1 \
    --code_dir ../code \
    --exp_dir $output_dir \
    --save_model --seed 1260 --consider_rst 1000 \
    --retrieve_learning --load_pre_trained_model --exp_dir_controller $pretrained_ctl_dir --model_name_controller $pretrained_ctl_name

    task2_pre_m=$output_dir/model_last.pth
done