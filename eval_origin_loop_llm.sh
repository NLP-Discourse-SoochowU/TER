cuda_id=0
bleurt_path=/home/longyin/bleurt/bleurt-large-512

type_=retrieve_learning_v6

# for round in {1..5};
# do
# prediction_file_path=exp/Controller_task1/$type_/round_$round/reproduce_task1_llm/predict.tsv
# output_dir=output/$type_/out_task1_llm/round_$round
# CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer.py --task "task_1" --split test --prediction_file $prediction_file_path --output_dir  $output_dir  --bleurt_checkpoint $bleurt_path
# done

# for round in {1..5};
# do
# prediction_file_path=exp/Controller_task2/$type_/round_$round/reproduce_task2_llm/predict.tsv
# output_dir=output/$type_/out_task2_llm/round_$round
# CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer.py --task "task_2" --split test --prediction_file $prediction_file_path --output_dir  $output_dir  --bleurt_checkpoint $bleurt_path
# done

for round in {5..5};
do
prediction_file_path=exp/Controller_task2/$type_/round_$round/reproduce_task3_llm/predict.tsv
output_dir=output/$type_/out_task3_llm/round_$round
CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer_task3.py --split test --prediction_file $prediction_file_path --output_dir  $output_dir --bleurt_checkpoint $bleurt_path
done