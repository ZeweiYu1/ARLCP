set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

train_files="./data/train/preprocessed_data/deepscaler.parquet"
val_files="['./data/test/preprocessed_data/gsm8k.parquet','./data/test/preprocessed_data/math.parquet','./data/test/preprocessed_data/aime24*16.parquet', './data/test/preprocessed_data/aime25*16.parquet','./data/test/preprocessed_data/amc23*16.parquet']"

MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # path to your download HF model

MODEL_PATH="/input2/yzw/models/DeepSeek-R1-Distill-Qwen-7B"

n_rollout=16

max_response_length=16384
LR=2e-6

PROJECT_NAME="ARLCP"
EXP_NAME="ARLCP-DeepSeek-R1-Distill-Qwen-7B"
CKPT_DIR="./ckpts/${PROJECT_NAME}/${EXP_NAME}"
TEST_DIR="./tests/${PROJECT_NAME}/${EXP_NAME}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.default_local_dir="${CKPT_DIR}" \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.max_critic_ckpt_to_keep=2 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    +trainer.validation_data_dir="${TEST_DIR}" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    custom_reward_function.path="./src/rm.py" \
    custom_reward_function.name="arlcp_rm" \
    trainer.total_epochs=1 $@ 2>&1 | tee ./log/ARLCP-7B.log