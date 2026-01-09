#!/bin/bash
# 使用 GRPO + 在线蒸馏训练搜索 agent
# 教师模型：Qwen3-30B-A3B-Instruct-2507
# 学生模型：Qwen3-4B-Instruct-2507

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_API_KEY="0559d52399bc5d3fd8e373bb4b8b6e8db054b9f7"

# 教师模型服务地址
# 需要先启动教师模型服务：
# vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --port 15555 \
#     --tensor-parallel-size 2 \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 32768 \
#     --trust-remote-code

TEACHER_BASE_URL="http://localhost:15555/v1"
TEACHER_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# 运行训练脚本
python3 -m examples.search.train_search_distill \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.val_batch_size=400 \
    data.max_prompt_length=2048 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=23000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    rllm.agent.max_steps=10 \
    rllm.distill.enable=True \
    rllm.distill.shared_tokenizer=False \
    rllm.distill.teacher_rollout_args.model="${TEACHER_MODEL}" \
    rllm.distill.teacher_rollout_args.base_url="${TEACHER_BASE_URL}" \
    rllm.distill.teacher_rollout_args.api_key="EMPTY" \
    rllm.distill.teacher_rollout_args.max_prompt_length=32768 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-search-agent' \
    trainer.experiment_name='qwen3-4b-grpo-distill-search-asearcher' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=50 \
    +retrieval_service_url="http://10.244.209.173:8000/retrieve" \
    +search_topk=5 \
    +search_timeout=60 \
    +parser_name=qwen

# 训练完成后清理
pkill -9 -f 'ray::WorkerDict'
