cuda_id=2
bleurt_path=/home/longyin/bleurt/bleurt-large-512

# Task1 Final
prediction_file_path=exp/Controller_task1/rhetorical/reproduce_task1_llm/predict.tsv
output_dir=output/rhetorical/out_task1_llm_bin
CUDA_VISIBLE_DEVICES=$cuda_id python eval/run_scorer_bin.py --task "task_1" --split test --prediction_file $prediction_file_path --output_dir  $output_dir  --bleurt_checkpoint $bleurt_path
