#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=12:00:00
#SBATCH --job-name=data_ablation_multimodal
#SBATCH --output=/iopsstor/scratch/cscs/%u/slurmlogs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/slurmlogs/%x-%j.err
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

echo "START TIME: $(date)"

################ Configs ################
DATA_DIR=/iopsstor/scratch/cscs/aditikhandelwal/pretraining-datasets

MBS=${OVERRIDE_MBS:-4}
GBS=${OVERRIDE_GBS:-256}         # 2x BS (fixed for data ablations)
TP=${OVERRIDE_TP:-1}
EP=${OVERRIDE_EP:-2}
PP=1
HIDDEN_SIZE=${OVERRIDE_HIDDEN_SIZE:-384}
MOE_FFN_HIDDEN_SIZE=${OVERRIDE_MOE_FFN_HIDDEN_SIZE:-$HIDDEN_SIZE}
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=${OVERRIDE_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-$HIDDEN_SIZE}
MOE_EXPERT_CAPACITY_FACTOR=${OVERRIDE_MOE_EXPERT_CAPACITY_FACTOR:-}
SEQ_LEN=4096
TOTAL_TOKENS=${OVERRIDE_TOTAL_TOKENS:-100000000000}  # 100B default
TOKENS_PER_STEP=$((GBS * SEQ_LEN))
TRAINING_STEPS=$((TOTAL_TOKENS / TOKENS_PER_STEP))
WARMUP_STEPS=$(( (TRAINING_STEPS * 3 + 99) / 100 ))
COOLDOWN_STEPS=$(( (TRAINING_STEPS * 10 + 99) / 100 ))
CHECKPOINT_STEPS=${OVERRIDE_CHECKPOINT_STEPS:-1000}

AUTO_JOB_REQUEUE=${OVERRIDE_AUTO_REQUEUE:-false}
SKIP_VALIDATION=${OVERRIDE_SKIP_VALIDATION:-false}

#### Debugging ####
LOG_NCCL=false
NSYS_PROFILER=false
MOCK_DATA=false
###################

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/megatron_trials/Megatron-LM
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache
BACKUP_CODEBASE=false

PROJECT_NAME=MultimodalScalingLawsV2
MODEL_NAME=${OVERRIDE_MODEL_NAME:-model2}
ABLATION_TAG=${OVERRIDE_TAG:-""}
EXP_NAME=${MODEL_NAME}-efficient-data-ablation-${SLURM_NNODES}n-${SEQ_LEN}sl-${GBS}gbsz${ABLATION_TAG:+-$ABLATION_TAG}
PROJECT_DIR=/iopsstor/scratch/cscs/aditikhandelwal/logs/$PROJECT_NAME

#########################################

EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
BACKUP_CODEBASE_DIR=$EXP_DIR/Megatron-LM

# Set up ENV
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=25680
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

#### Megatron Args ####
TRANSFORMER_ENGINE_ARGS=(
	--transformer-impl transformer_engine
	--use-precision-aware-optimizer
	--main-grads-dtype bf16
)

NETWORK_SIZE_ARGS=(
	--num-layers 32
	--hidden-size $HIDDEN_SIZE
	--num-attention-heads 16
	--group-query-attention
	--untie-embeddings-and-output-weights
	--num-query-groups 4
	--max-position-embeddings $SEQ_LEN
	--position-embedding-type rope
	--rotary-base 500000
	--use-rope-scaling
	--rope-scaling-factor 32
	--make-vocab-size-divisible-by 128
	--normalization RMSNorm
	--swiglu
	--qk-layernorm
	--qknorm-impl apex
)

LOGGING_ARGS=(
	--log-throughput
	--log-progress
	--tensorboard-dir $TENSORBOARD_DIR
	--no-log-loss-scale-to-tensorboard
	--log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--weight-decay 0.1
	--clip-grad 1.0
	--adam-beta1 0.9
	--adam-beta2 0.95
)

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--no-check-for-nan-in-loss-and-grad
	--train-iters $TRAINING_STEPS
	--log-interval 1
	--cross-entropy-loss-fusion
	--disable-bias-linear
	--optimizer adam
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 500
	--eval-interval 100000000000
    --eval-iters 0
	--trigger-path $TRIGGER_DIR
)

INITIALIZATION_ARGS=(
	--seed ${OVERRIDE_SEED:-28}
	--init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
	--lr ${OVERRIDE_LR:-0.00300541}
	--min-lr ${OVERRIDE_MIN_LR:-0.000300541}
	--lr-decay-style WSD
	--lr-warmup-iters $WARMUP_STEPS
	--lr-wsd-decay-style linear
	--lr-wsd-decay-iters $COOLDOWN_STEPS
)

CHECKPOINTING_ARGS=(
	--save $CKPT_DIR
	--save-interval $CHECKPOINT_STEPS
	--ckpt-format torch_dist
	--load $CKPT_DIR
	--async-save
)

MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP
	--expert-model-parallel-size $EP
	--use-distributed-optimizer
	--overlap-grad-reduce
	--overlap-param-gather
	--sequence-parallel
)

MOE_ARGS=(
    --num-experts 16
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
	--moe-token-dispatcher-type allgather
	--moe-router-fusion
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
)

if [ -n "$MOE_EXPERT_CAPACITY_FACTOR" ]; then
    MOE_ARGS+=(--moe-expert-capacity-factor "$MOE_EXPERT_CAPACITY_FACTOR")
fi

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok
)

DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
	--reset-position-ids
	--reset-attention-mask
	--eod-mask-loss
	--num-workers 4
	--num-dataset-builder-threads 32
)

# Set up directories
mkdir -p $CKPT_DIR
mkdir -p $PROJECT_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR

# Backup codebase
if [ "$BACKUP_CODEBASE" == true ]; then
  if [ -z "$(ls -A "$BACKUP_CODEBASE_DIR")" ]; then
  	echo "[$(date)] Copying codebase in $MEGATRON_LM_DIR to $BACKUP_CODEBASE_DIR..."
  	rsync -av --exclude-from=$MEGATRON_LM_DIR/.gitignore $MEGATRON_LM_DIR/ $BACKUP_CODEBASE_DIR/ &> /dev/null
  fi
  MEGATRON_LM_DIR=$BACKUP_CODEBASE_DIR
fi

echo "[$(date)] Using codebase in $MEGATRON_LM_DIR"

cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# Data path — overridable via OVERRIDE_DATA_PATH env var
if [ -n "$OVERRIDE_DATA_PATH" ]; then
  WEIGHTED_DATA_PATH="$OVERRIDE_DATA_PATH"
else
  # Default: text 66%, audio_only 8.5%, audio_text 8.5%, vision_img2txt 8.5%, vision_txt2img 8.5%
  WEIGHTED_DATA_PATH="\
    0.66 $DATA_DIR/merged/text_only_merged \
    0.085 $DATA_DIR/audio_only/train/merged \
    0.085 $DATA_DIR/audio_text_interleaved/train/MERGED_0.1_36B/merged \
    0.085 $DATA_DIR/vision_text_interleaved/train_img_conditioned_on_text/llavaOv1_5_Midtrain_paired_apertus8b_emu3p5_merged \
    0.085 $DATA_DIR/vision_text_interleaved/train_text_conditioned_on_img/llavaOv1_5_Midtrain_paired_apertus8b_emu3p5_merged"
fi

if [ "$MOCK_DATA" = true ]; then
  DATA_ARGS="${DATA_ARGS[@]} --mock-data"
else
  DATA_ARGS="${DATA_ARGS[@]} --data-path $WEIGHTED_DATA_PATH --data-cache-path $DATASET_CACHE_DIR"
fi

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
	${MOE_ARGS[@]} \
    $DATA_ARGS"

# Hugging Face Token
export HF_TOKEN=''
export WANDB_API_KEY=''
export TRANSFORMERS_NO_SLOW_TOKENIZER=1

if [ -n "$WANDB_API_KEY" ]; then
  echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
  if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
    echo "[$(date)] Syncing WANDB from previous run"
    wandb sync "$LOGGING_DIR/wandb/latest-run"
  fi
  TRAINING_CMD="$TRAINING_CMD \
    --wandb-save-dir $LOGGING_DIR \
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
  export WANDB_MODE=disabled
  echo "[$(date)] No WANDB API key found. WANDB logging disabled."
fi

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

# NSYS profiler
if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace-hostname-\$SLURMD_NODENAME-procid-\$SLURM_PROCID.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi

# Save sbatch script
cp $0 $DEBUG_DIR

# Clean triggers
rm -f $TRIGGER_DIR/save
rm -f $TRIGGER_DIR/exit

# Checkpoint Compute Environment
echo -e "$(date)" > $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nCMD: $CMD_PREFIX $TRAINING_CMD" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nSlurm file: $0\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $0 >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nTOML file: $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nNODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nMegatron path: $MEGATRON_LM_DIR ($(git -C $MEGATRON_LM_DIR rev-parse --verify HEAD))" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\n$(pip list)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\n$(nvidia-smi)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nEnvironment Variables:\n\n$(printenv)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR

# before you call python / srun
export LOCAL_CACHE_BASE=${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}
export TRITON_CACHE_DIR=${LOCAL_CACHE_BASE}/triton_cache/${SLURM_PROCID}
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_CACHE_BASE}/inductor_cache/${SLURM_PROCID}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

if [ "$AUTO_JOB_REQUEUE" = true ]; then
	echo "[$(date)] $(OVERRIDE_TOTAL_TOKENS=${OVERRIDE_TOTAL_TOKENS} OVERRIDE_DATA_PATH=\"${OVERRIDE_DATA_PATH}\" OVERRIDE_MBS=${OVERRIDE_MBS} OVERRIDE_GBS=${OVERRIDE_GBS} OVERRIDE_TP=${OVERRIDE_TP} OVERRIDE_EP=${OVERRIDE_EP} OVERRIDE_HIDDEN_SIZE=${OVERRIDE_HIDDEN_SIZE} OVERRIDE_MOE_FFN_HIDDEN_SIZE=${OVERRIDE_MOE_FFN_HIDDEN_SIZE} OVERRIDE_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=${OVERRIDE_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE} OVERRIDE_MOE_EXPERT_CAPACITY_FACTOR=${OVERRIDE_MOE_EXPERT_CAPACITY_FACTOR} OVERRIDE_LR=${OVERRIDE_LR} OVERRIDE_MIN_LR=${OVERRIDE_MIN_LR} OVERRIDE_MODEL_NAME=${OVERRIDE_MODEL_NAME} OVERRIDE_CHECKPOINT_STEPS=${OVERRIDE_CHECKPOINT_STEPS} OVERRIDE_SEED=${OVERRIDE_SEED} OVERRIDE_TAG=${OVERRIDE_TAG} sbatch --dependency=singleton $0)"
fi

srun --cpus-per-task "$SLURM_CPUS_PER_TASK" --mpi=pmix \
  --distribution=block:block \
  --network=disable_rdzv_get \
  --environment=/iopsstor/scratch/cscs/aditikhandelwal/megatron_trials/Megatron-LM/tomls/new_toml_alps3.toml \
  -lu bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"

TRAINING_COMPLETE=false
if [ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
  LAST_CKPT_ITER=$(tr -d '[:space:]' < "$CKPT_DIR/latest_checkpointed_iteration.txt")
  if [[ "$LAST_CKPT_ITER" =~ ^[0-9]+$ ]] && [ "$LAST_CKPT_ITER" -ge "$TRAINING_STEPS" ]; then
    TRAINING_COMPLETE=true
  fi
fi

if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -rf $TRIGGER_DIR/exit
   scancel --jobname $SLURM_JOB_NAME
fi

# Run validation evaluation on the final checkpoint unless disabled for short sweeps
if [ "$SKIP_VALIDATION" = true ]; then
  echo "[$(date)] Skipping validation submission."
elif [ "$TRAINING_COMPLETE" != true ]; then
  echo "[$(date)] Training is not complete yet; skipping validation submission."
else
  echo "[$(date)] Submitting validation evaluation job..."
  CKPT_DIR=$CKPT_DIR VAL_SETS_DIR=/iopsstor/scratch/cscs/aditikhandelwal/datasets/validation VAL_MAX_ITERS=50 sbatch $MEGATRON_LM_DIR/run_validation.sh
fi
