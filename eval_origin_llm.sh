cuda_id=2
bleurt_path=/home/longyin/bleurt/bleurt-large-512


# evaluate baseline system LLM
# prediction_file_path=exp/Controller_task1/z7SOg44r/reproduce_task1_llm_bin/predict.tsv
# output_dir=output/baseline/out_task1_llm_bin
# CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer_bin.py --task "task_1" --split test --prediction_file $prediction_file_path --output_dir  $output_dir  --bleurt_checkpoint $bleurt_path
# prediction_file_path=exp/Controller_task2/NuCkQlfx/reproduce_task2_llm_bin/predict.tsv
# output_dir=output/baseline/out_task2_llm_bin
# CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer.py --task "task_2" --split test --prediction_file $prediction_file_path --output_dir  $output_dir  --bleurt_checkpoint $bleurt_path
# prediction_file_path=exp/Controller_task2/NuCkQlfx/reproduce_task3_llm_bin/predict.tsv
# output_dir=output/baseline/out_task3_llm_bin
# CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer_task3.py --split test --prediction_file $prediction_file_path --output_dir  $output_dir --bleurt_checkpoint $bleurt_path

# For rhetorical purpose LLM 
prediction_file_path=exp/Controller_task1/rhetorical/reproduce_task1_llama/predict.tsv
output_dir=output/rhetorical/out_task1_llama
CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer.py --task "task_1" --split test --prediction_file $prediction_file_path --output_dir  $output_dir  --bleurt_checkpoint $bleurt_path
# prediction_file_path=exp/Controller_task2/rhetorical/reproduce_task2_llm/predict.tsv
# output_dir=output/rhetorical/out_task2_llm
# CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer.py --task "task_2" --split test --prediction_file $prediction_file_path --output_dir  $output_dir  --bleurt_checkpoint $bleurt_path
# prediction_file_path=exp/Controller_task2/rhetorical/reproduce_task3_llm/predict.tsv
# output_dir=output/rhetorical/out_task3_llm
# CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer_task3.py --split test --prediction_file $prediction_file_path --output_dir  $output_dir --bleurt_checkpoint $bleurt_path
