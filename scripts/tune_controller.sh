cuda_id=1

cd ../code
data_dir=../data/Controller_data

# Make controller training data based on the orginal dataset and the trained module
# modify the arguments in make_controller_data.py and run it
# CUDA_VISIBLE_DEVICES=0  python make_controller_data.py

# # Task 1
# pretrained_ctl_dir=../exp/Controller_task1/z7SOg44r
# pretrained_ctl_name=model_task1.pth
# CUDA_VISIBLE_DEVICES=$cuda_id python train_Controller.py \
# --task_name task1 \
# --train_data_file $data_dir/train.controller.task1.v36.jsonl \
# --dev_data_file $data_dir/dev.controller.task1.v36.jsonl \
# --model_name_or_path albert-xxlarge-v2 \
# --bs 6 --lr 1e-5 --epochs 50 --adafactor \
# --eval_epoch 2 --report_epoch 1 \
# --code_dir ../code \
# --exp_dir ../exp/Controller_task1/rhetorical \
# --save_model --seed 2171 --consider_rst 1 \
# --load_pre_trained_model --exp_dir_controller $pretrained_ctl_dir --model_name_controller $pretrained_ctl_name


# Task 2 & 3
pretrained_ctl_dir=../exp/Controller_task2/NuCkQlfx
pretrained_ctl_name=model_task2.pth
CUDA_VISIBLE_DEVICES=$cuda_id python train_Controller.py \
--task_name task2 \
--train_data_file $data_dir/train.controller.task2.v36.jsonl \
--dev_data_file $data_dir/dev.controller.task2.v36.jsonl \
--model_name_or_path albert-xxlarge-v2 \
--bs 6 --lr 1e-5 --epochs 50 --adafactor \
--eval_epoch 2 --report_epoch 1 \
--code_dir ../code \
--exp_dir ../exp/Controller_task2/rhetorical \
--save_model --seed 1260  --consider_rst 1 \
--load_pre_trained_model --exp_dir_controller $pretrained_ctl_dir --model_name_controller $pretrained_ctl_name