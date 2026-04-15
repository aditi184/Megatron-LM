#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=12:00:00
#SBATCH --job-name=val_multimodal
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
# Path to the checkpoint to evaluate (should be the same --save dir used during training)
CKPT_DIR="${CKPT_DIR:-}"

# Path to per-modality validation sets (created by create_validation_sets.py)
VAL_SETS_DIR="${VAL_SETS_DIR:-/iopsstor/scratch/cscs/aditikhandelwal/datasets/validation}"

# Where to save evaluation results JSON
VAL_RESULTS_FILE="${eval_results.json}"

# Max eval iters per validation set (empty = eval all data)
VAL_MAX_ITERS="${VAL_MAX_ITERS:-}"

# Evaluate all checkpoints (set to "true" to enable)
VAL_ALL_CKPTS="${VAL_ALL_CKPTS:-true}"

# Model configs (must match training) — override via env vars
MBS="${VAL_MBS:-4}"
GBS="${VAL_GBS:-1024}"
TP="${VAL_TP:-2}"
EP="${VAL_EP:-1}"
PP="${VAL_PP:-1}"
SEQ_LEN="${VAL_SEQ_LEN:-4096}"
HIDDEN_SIZE="${VAL_HIDDEN_SIZE:-384}"
MOE_FFN_HIDDEN="${VAL_MOE_FFN_HIDDEN:-384}"
MOE_SHARED_EXPERT="${VAL_MOE_SHARED_EXPERT:-384}"
MOE_DISPATCHER="${VAL_MOE_DISPATCHER:-allgather}"
LR="${VAL_LR:-0.00300541}"
MIN_LR="${VAL_MIN_LR:-0.000300541}"
LR_WARMUP="${VAL_LR_WARMUP:-300}"

# W&B: reuse the training run's logging dir to resume logging
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/megatron_trials/Megatron-LM
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache

PROJECT_NAME=MultimodalScalingLawsV2
EXP_NAME="${VAL_EXP_NAME:-model2-ablation-16n-${SEQ_LEN}sl-${GBS}gbsz-lr1.0x-bs1.0x-s28}"

PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME
EXP_DIR=$PROJECT_DIR/$EXP_NAME
LOGGING_DIR="${VAL_LOGGING_DIR:-$EXP_DIR/logging}"
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
#########################################

# Set up ENV
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=300

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=25681
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

#### Megatron Args (must match training) ####
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
	--train-iters 0
	--log-interval 1
	--cross-entropy-loss-fusion
	--disable-bias-linear
	--optimizer adam
	--dataloader-type single
	--eval-interval 100000000000
	--eval-iters 0
	--skip-train
)

INITIALIZATION_ARGS=(
	--seed 28
	--init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
	--lr $LR
	--min-lr $MIN_LR
	--lr-decay-style WSD
	--lr-warmup-iters $LR_WARMUP
	--lr-wsd-decay-style linear
	--lr-wsd-decay-iters 1000
)

CHECKPOINTING_ARGS=(
	--load $CKPT_DIR
	--ckpt-format torch_dist
)

MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP
	--expert-model-parallel-size $EP
	--sequence-parallel
)

MOE_ARGS=(
	--num-experts 16
	--moe-router-load-balancing-type aux_loss
	--moe-router-topk 2
	--moe-aux-loss-coeff 1e-2
	--moe-grouped-gemm
	--moe-token-dispatcher-type $MOE_DISPATCHER
	--moe-router-fusion
	--moe-ffn-hidden-size $MOE_FFN_HIDDEN
	--moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT
)

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
	--data-cache-path $DATASET_CACHE_DIR
	--mock-data
)

VALIDATION_ARGS=(
	--val-sets-dir $VAL_SETS_DIR
	--val-results-file $VAL_RESULTS_FILE
)

# Add --val-max-iters only if set
if [ -n "$VAL_MAX_ITERS" ]; then
	VALIDATION_ARGS+=(--val-max-iters $VAL_MAX_ITERS)
fi

# Add --val-all-checkpoints if enabled
if [ "$VAL_ALL_CKPTS" = "true" ]; then
	VALIDATION_ARGS+=(--val-all-checkpoints)
fi

# Evaluate 1/3 of checkpoints evenly spaced (override via VAL_FRACTION env var)
VAL_FRACTION="${VAL_FRACTION:-0.33}"
if [ -n "$VAL_FRACTION" ]; then
	VALIDATION_ARGS+=(--val-fraction $VAL_FRACTION)
fi

cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

export HF_TOKEN=''
export WANDB_API_KEY='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'
export TRANSFORMERS_NO_SLOW_TOKENIZER=1

EVAL_CMD="python3 $MEGATRON_LM_DIR/evaluate_validation.py \
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
	${DATA_ARGS[@]} \
	${VALIDATION_ARGS[@]}"

# Disable W&B for validation — wandb.init() times out on compute nodes
# and crashes the rank, preventing distributed init from completing.
# Results are saved to the JSON file instead.
export WANDB_MODE=disabled
echo "[$(date)] W&B disabled for validation. Results will be saved to JSON."

# Cache directories
export LOCAL_CACHE_BASE=${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}
export TRITON_CACHE_DIR=${LOCAL_CACHE_BASE}/triton_cache/${SLURM_PROCID}
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_CACHE_BASE}/inductor_cache/${SLURM_PROCID}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "[$(date)] Running validation evaluation..."
echo "[$(date)] Checkpoint: $CKPT_DIR"
echo "[$(date)] Validation sets: $VAL_SETS_DIR"
echo "[$(date)] Results file: $VAL_RESULTS_FILE"

srun --cpus-per-task "$SLURM_CPUS_PER_TASK" --mpi=pmix \
	--distribution=block:block \
	--network=disable_rdzv_get \
	--environment=/iopsstor/scratch/cscs/aditikhandelwal/megatron_trials/Megatron-LM/tomls/new_toml_alps3.toml \
	-lu bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID numactl --membind=0-3 $EVAL_CMD"

SRUN_EXIT=$?
echo "END TIME: $(date)"
if [ $SRUN_EXIT -eq 0 ]; then
	echo "Results saved to: $VAL_RESULTS_FILE"
else
	echo "ERROR: Validation failed with exit code $SRUN_EXIT"
fi
