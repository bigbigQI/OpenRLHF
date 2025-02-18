
HDFS_HOME=.
RUN_NAME=Qwen2.5_1.5B_distill_rloo

python3 openrlhf/cli/train_ppo_ray.py \
    --advantage_estimator rloo \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_path $HDFS_HOME/checkpoints/$RUN_NAME \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 128 \
    --temperature 0.8 \
    --n_samples_per_prompt 8 \
    --max_samples 100000 \
    --max_epochs 1 \
    --num_episodes 20 \
    --prompt_max_len 1024 \
    --generate_max_len 14000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 2e-6 \
    --init_kl_coef 0.001 \
    --prompt_data pe-nlp/DeepScaleR-40k-Prompt-NoSys \
    --input_key input \
    --normalize_reward \
    --flash_attn \
    --adam_offload \
    --gradient_checkpointing \
    --save_steps 10 \
    --load_checkpoint \
    --use_wandb 149737fd3c4537b349a37aab90b6fff96f385ebc \
    --wandb_run_name $RUN_NAME \
    --wandb_project rl_from_distill_math \
    --ckpt_path $HDFS_HOME/checkpoints/$RUN_NAME  \
    --max_ckpt_num 5 \
    --remote_rm_url examples/scripts/r1_reward_func.py