# 1. multi-task learning; 2. learned label vectors.

cd ../code
data_dir=../data/Controller_data


#### Task 1
## Step1: make controller training data based on the orginal dataset and the trained module
# modify the arguments in make_controller_data.py and run it
#CUDA_VISIBLE_DEVICES=0  python make_controller_data.py

### Step2: train the controller
# consider_rst 10, 500, 100000000000
# pretrained_ctl_dir=../exp/Controller_task1/xxx
# iter=5252
# pretrained_ctl_name=model_$iter.pth

# CUDA_VISIBLE_DEVICES=0 python train_Controller.py \
# --task_name task1 \
# --train_data_file $data_dir/train.controller.task1.v36.jsonl \
# --dev_data_file $data_dir/dev.controller.task1.v36.jsonl \
# --model_name_or_path albert-xxlarge-v2 \
# --bs 6 --lr 1e-5 --epochs 1000 --adafactor \
# --eval_epoch 2 --report_epoch 1 \
# --code_dir ../code \
# --exp_dir ../exp/Controller_task1/xxx \
# --save_model --seed 2171 \
# --load_pre_trained_model --exp_dir_controller $pretrained_ctl_dir --model_name_controller $pretrained_ctl_name


##### Task 2 & Task 3
### Step1: make controller training data based on the orginal dataset and the trained module
# modify the arguments in make_controller_data.py and run it
#CUDA_VISIBLE_DEVICES=3  python make_controller_data.py

### Step2: train the controller
# pretrained_ctl_dir=../exp/Controller_task2/xxx
# iter=2626
# pretrained_ctl_name=model_$iter.pth
# CUDA_VISIBLE_DEVICES=1 python train_Controller.py \
# --task_name task2 \
# --train_data_file $data_dir/train.controller.task2.v36.jsonl \
# --dev_data_file $data_dir/dev.controller.task2.v36.jsonl \
# --model_name_or_path albert-xxlarge-v2 \
# --bs 6 --lr 1e-5 --epochs 1000 --adafactor \
# --eval_epoch 2 --report_epoch 1 \
# --code_dir ../code \
# --exp_dir ../exp/Controller_task2/xxx \
# --save_model --seed 1260 \
# --load_pre_trained_model --exp_dir_controller $pretrained_ctl_dir --model_name_controller $pretrained_ctl_name