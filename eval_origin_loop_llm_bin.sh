cuda_id=0
bleurt_path=/home/longyin/bleurt/bleurt-large-512

type_=retrieve_learning_v6

# Task2 Final

# for round in {1..5};
# do
# prediction_file_path=exp/Controller_task2/$type_/round_$round/reproduce_task2_llm/predict.tsv
# output_dir=output/$type_/out_task2_llm_bin/round_$round
# CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer_bin.py --task "task_2" --split test --prediction_file $prediction_file_path --output_dir  $output_dir  --bleurt_checkpoint $bleurt_path
# done

# Task3 Final
for round in {5..5};
do
prediction_file_path=exp/Controller_task2/$type_/round_$round/reproduce_task3_llm/predict.tsv
output_dir=output/$type_/out_task3_llm_bin/round_$round
CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer_task3_bin.py --split test --prediction_file $prediction_file_path --output_dir  $output_dir --bleurt_checkpoint $bleurt_path
done