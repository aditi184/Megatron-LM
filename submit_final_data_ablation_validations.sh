#!/bin/bash
#
# Submit final-checkpoint validation jobs for completed data-ablation runs.
#
# This script:
# - reconstructs the expected data-ablation experiment names from hyperparamters.csv
# - only submits validation for runs whose latest checkpoint reaches the target steps
# - evaluates only the final checkpoint
# - evaluates all samples in each validation set
# - skips runs whose final checkpoint already appears in validation results JSON
# - writes a CSV manifest summarizing status for every expected run
#
# Usage:
#   bash submit_final_data_ablation_validations.sh
#   bash submit_final_data_ablation_validations.sh --models "Model 7"
#   bash submit_final_data_ablation_validations.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_LM_DIR="$SCRIPT_DIR"
RUN_VALIDATION_SCRIPT="$MEGATRON_LM_DIR/run_validation.sh"
HYPERPARAMS_CSV="$SCRIPT_DIR/hyperparamters.csv"

DATA_DIR=/iopsstor/scratch/cscs/aditikhandelwal/pretraining-datasets
PROJECT_NAME=MultimodalDataAblationScalingLaws
LOGS_BASE=/iopsstor/scratch/cscs/aditikhandelwal/logs
VAL_SETS_DIR=/iopsstor/scratch/cscs/aditikhandelwal/datasets/validation
MANIFEST_PATH="$LOGS_BASE/$PROJECT_NAME/final_validation_manifest.csv"

TEXT_PATH="$DATA_DIR/merged/text_only_merged"
VISION_I2T_PATH="$DATA_DIR/vision_text_interleaved/train_img_conditioned_on_text/llavaOv1_5_Midtrain_paired_apertus8b_emu3p5_merged"
VISION_T2I_PATH="$DATA_DIR/vision_text_interleaved/train_text_conditioned_on_img/llavaOv1_5_Midtrain_paired_apertus8b_emu3p5_merged"
AUDIO_ONLY_PATH="$DATA_DIR/audio_only/train/merged"
AUDIO_TEXT_PATH="$DATA_DIR/audio_text_interleaved/train/MERGED_0.1_36B/merged"

MODAL_TOKENS_B=(5 10 15 20 25)
INTERFERENCE_SPLITS=("45:5" "41.5:8.5" "37.5:12.5" "33:17" "25:25")

SEQ_LEN=4096
MODEL_FILTER=""
DRY_RUN=false
RESERVATION=""
WALLTIME=""

declare -A HP_TP HP_EP HP_MBS HP_GBS HP_LR HP_HS
MODELS=()

usage() {
    sed -n '1,18p' "$0"
}

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    echo "$s"
}

normalize_model() {
    local lower
    lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    echo "$lower" | tr -d '[:space:]'
}

to_tokens() {
    python3 -c "print(int(float('$1') * 1000000000))"
}

weight() {
    python3 -c "print(f'{float(\"$1\") / float(\"$2\"):.6f}')"
}

min_lr_for() {
    python3 -c "print(f'{float(\"$1\") / 10.0:.10g}')"
}

add_model_once() {
    local model="$1"
    local existing
    for existing in "${MODELS[@]}"; do
        if [ "$existing" = "$model" ]; then
            return
        fi
    done
    MODELS+=("$model")
}

load_hyperparams() {
    local model tp ep mbs gbs lr datasize hs key

    if [ ! -f "$HYPERPARAMS_CSV" ]; then
        echo "Missing hyperparameter CSV: $HYPERPARAMS_CSV" >&2
        exit 1
    fi

    while IFS='|' read -r model tp ep mbs gbs lr datasize hs; do
        key="$model|$datasize"
        HP_TP["$key"]="$tp"
        HP_EP["$key"]="$ep"
        HP_MBS["$key"]="$mbs"
        HP_GBS["$key"]="$gbs"
        HP_LR["$key"]="$lr"
        HP_HS["$key"]="$hs"
        add_model_once "$model"
    done < <(python3 - "$HYPERPARAMS_CSV" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        cells = [cell.strip() for cell in row]
        if not cells or not cells[0] or cells[0].lower() == "model":
            continue
        if len(cells) < 8:
            continue
        print("|".join(cells[:8]))
PY
    )
}

model_selected() {
    local model="$1"
    local wanted

    if [ -z "$MODEL_FILTER" ]; then
        return 0
    fi

    IFS=',' read -r -a wanted_models <<< "$MODEL_FILTER"
    for wanted in "${wanted_models[@]}"; do
        wanted="$(trim "$wanted")"
        if [ "$(normalize_model "$wanted")" = "$(normalize_model "$model")" ]; then
            return 0
        fi
    done

    return 1
}

append_case() {
    CASE_TAGS+=("$1")
    CASE_DATASIZES+=("$2")
    CASE_TOTAL_TOKENS+=("$3")
    CASE_DATA_PATHS+=("$4")
}

build_cases() {
    CASE_TAGS=()
    CASE_DATASIZES=()
    CASE_TOTAL_TOKENS=()
    CASE_DATA_PATHS=()

    local vb ab split vis_b aud_b text_w vis_w aud_w data_path

    for vb in "${MODAL_TOKENS_B[@]}"; do
        text_w=$(weight "$(python3 -c "print(100.0 - float('$vb'))")" 100)
        vis_w=$(weight "$vb" 200)
        data_path="$text_w $TEXT_PATH $vis_w $VISION_I2T_PATH $vis_w $VISION_T2I_PATH"
        append_case "text-vision-${vb}Bvis" 100 100000000000 "$data_path"
    done

    for vb in "${MODAL_TOKENS_B[@]}"; do
        data_path="0.5 $VISION_I2T_PATH 0.5 $VISION_T2I_PATH"
        append_case "vision-only-${vb}B" "$vb" "$(to_tokens "$vb")" "$data_path"
    done

    for ab in "${MODAL_TOKENS_B[@]}"; do
        text_w=$(weight "$(python3 -c "print(100.0 - float('$ab'))")" 100)
        aud_w=$(weight "$ab" 200)
        data_path="$text_w $TEXT_PATH $aud_w $AUDIO_ONLY_PATH $aud_w $AUDIO_TEXT_PATH"
        append_case "text-audio-${ab}Baud" 100 100000000000 "$data_path"
    done

    for ab in "${MODAL_TOKENS_B[@]}"; do
        data_path="0.5 $AUDIO_ONLY_PATH 0.5 $AUDIO_TEXT_PATH"
        append_case "audio-only-${ab}B" "$ab" "$(to_tokens "$ab")" "$data_path"
    done

    for split in "${INTERFERENCE_SPLITS[@]}"; do
        vis_b="${split%%:*}"
        aud_b="${split##*:}"
        text_w=$(weight 100 150)
        vis_w=$(weight "$vis_b" 300)
        aud_w=$(weight "$aud_b" 300)
        data_path="$text_w $TEXT_PATH $vis_w $VISION_I2T_PATH $vis_w $VISION_T2I_PATH $aud_w $AUDIO_ONLY_PATH $aud_w $AUDIO_TEXT_PATH"
        append_case "interference-${vis_b}Bvis-${aud_b}Baud" 150 150000000000 "$data_path"
    done
}

json_has_result_for_iteration() {
    local json_path="$1"
    local ckpt_dir="$2"
    local iteration="$3"

    if [ ! -f "$json_path" ]; then
        return 1
    fi

    python3 - "$json_path" "$ckpt_dir" "$iteration" <<'PY'
import json
import sys

json_path, ckpt_dir, iteration = sys.argv[1:]
key = f"{ckpt_dir}::iter{iteration}"

try:
    with open(json_path) as f:
        data = json.load(f)
except Exception:
    sys.exit(1)

entry = data.get(key)
if isinstance(entry, dict):
    sys.exit(0)
sys.exit(1)
PY
}

csv_escape() {
    local s="${1:-}"
    s="${s//\"/\"\"}"
    printf '"%s"' "$s"
}

write_manifest_header() {
    mkdir -p "$(dirname "$MANIFEST_PATH")"
    printf '%s\n' \
        'model,case_tag,datasize_b,exp_name,exp_dir,ckpt_dir,target_steps,last_checkpoint,status,results_file,job_id' \
        > "$MANIFEST_PATH"
}

append_manifest_row() {
    local model="$1"
    local case_tag="$2"
    local datasize="$3"
    local exp_name="$4"
    local exp_dir="$5"
    local ckpt_dir="$6"
    local target_steps="$7"
    local last_ckpt="$8"
    local status="$9"
    local results_file="${10}"
    local job_id="${11}"

    {
        csv_escape "$model"; printf ','
        csv_escape "$case_tag"; printf ','
        csv_escape "$datasize"; printf ','
        csv_escape "$exp_name"; printf ','
        csv_escape "$exp_dir"; printf ','
        csv_escape "$ckpt_dir"; printf ','
        csv_escape "$target_steps"; printf ','
        csv_escape "$last_ckpt"; printf ','
        csv_escape "$status"; printf ','
        csv_escape "$results_file"; printf ','
        csv_escape "$job_id"; printf '\n'
    } >> "$MANIFEST_PATH"
}

shell_quote() {
    printf '%q' "$1"
}

append_batch_run() {
    local batch_script="$1"
    local exp_name="$2"
    local exp_dir="$3"
    local ckpt_dir="$4"
    local mbs="$5"
    local tp="$6"
    local ep="$7"
    local hs="$8"
    local lr="$9"
    local min_lr="${10}"
    local warmup_steps="${11}"
    local results_file="${12}"

    mkdir -p "$exp_dir/validation"

    {
        printf 'echo "[%s] Running %s"\n' '$(date)' "$(shell_quote "$exp_name")"
        printf 'CKPT_DIR=%s ' "$(shell_quote "$ckpt_dir")"
        printf 'VAL_SETS_DIR=%s ' "$(shell_quote "$VAL_SETS_DIR")"
        printf 'VAL_RESULTS_FILE=%s ' "$(shell_quote "$results_file")"
        printf 'VAL_MAX_ITERS= '
        printf 'VAL_ALL_CKPTS=false '
        printf 'VAL_FRACTION= '
        printf 'VAL_PROJECT_NAME=%s ' "$(shell_quote "$PROJECT_NAME")"
        printf 'VAL_EXP_NAME=%s ' "$(shell_quote "$exp_name")"
        printf 'VAL_LOGGING_DIR=%s ' "$(shell_quote "$exp_dir/logging")"
        printf 'VAL_MBS=%s ' "$(shell_quote "$mbs")"
        printf 'VAL_GBS=1024 '
        printf 'VAL_TP=%s ' "$(shell_quote "$tp")"
        printf 'VAL_EP=%s ' "$(shell_quote "$ep")"
        printf 'VAL_PP=1 '
        printf 'VAL_SEQ_LEN=%s ' "$(shell_quote "$SEQ_LEN")"
        printf 'VAL_HIDDEN_SIZE=%s ' "$(shell_quote "$hs")"
        printf 'VAL_MOE_FFN_HIDDEN=%s ' "$(shell_quote "$hs")"
        printf 'VAL_MOE_SHARED_EXPERT=%s ' "$(shell_quote "$hs")"
        printf 'VAL_MOE_DISPATCHER=allgather '
        printf 'VAL_LR=%s ' "$(shell_quote "$lr")"
        printf 'VAL_MIN_LR=%s ' "$(shell_quote "$min_lr")"
        printf 'VAL_LR_WARMUP=%s ' "$(shell_quote "$warmup_steps")"
        printf 'bash %s\n' "$(shell_quote "$RUN_VALIDATION_SCRIPT")"
        printf '\n'
    } >> "$batch_script"
}

submit_validation_batch() {
    local model="$1"
    local model_tag="$2"
    local batch_script="$3"
    local -a sbatch_args

    sbatch_args=(
        --parsable
        --account=infra01
        --nodes=16
        --ntasks-per-node=4
        --gpus-per-node=4
        --cpus-per-task=72
        --mem=460000
        --no-requeue
        --job-name="val_data_ablation_${model_tag}"
        --output="/iopsstor/scratch/cscs/%u/slurmlogs/val_data_ablation_${model_tag}-%j.out"
        --error="/iopsstor/scratch/cscs/%u/slurmlogs/val_data_ablation_${model_tag}-%j.err"
    )
    if [ -n "$WALLTIME" ]; then
        sbatch_args+=(--time="$WALLTIME")
    else
        sbatch_args+=(--time=12:00:00)
    fi
    if [ -n "$RESERVATION" ]; then
        sbatch_args+=(--reservation="$RESERVATION")
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  DRY RUN submit batch for $model using $batch_script"
        return 0
    fi

    sbatch "${sbatch_args[@]}" "$batch_script"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --models) MODEL_FILTER="$2"; shift 2 ;;
        --hyperparams-csv) HYPERPARAMS_CSV="$2"; shift 2 ;;
        --manifest) MANIFEST_PATH="$2"; shift 2 ;;
        --time) WALLTIME="$2"; shift 2 ;;
        --reservation) RESERVATION="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ ! -f "$RUN_VALIDATION_SCRIPT" ]; then
    echo "Missing validation script: $RUN_VALIDATION_SCRIPT" >&2
    exit 1
fi

load_hyperparams
build_cases
write_manifest_header

echo ""
echo "============================================================="
echo "  Final data-ablation validations"
echo "============================================================="
echo "  validation script=$RUN_VALIDATION_SCRIPT"
echo "  hyperparams=$HYPERPARAMS_CSV"
echo "  models=${MODEL_FILTER:-all from CSV}"
echo "  manifest=$MANIFEST_PATH"
echo "  final checkpoint only=true"
echo "  all validation samples=true"
if [ "$DRY_RUN" = true ]; then
    echo "  dry-run=true"
fi
echo ""

submitted=0
skipped_validated=0
skipped_incomplete=0
skipped_missing=0

for model in "${MODELS[@]}"; do
    if ! model_selected "$model"; then
        continue
    fi

    model_tag="$(normalize_model "$model")"
    echo "Model: $model"
    pending_count=0
    batch_job_id=""
    batch_script=""

    for i in "${!CASE_TAGS[@]}"; do
        case_tag="${CASE_TAGS[$i]}"
        datasize="${CASE_DATASIZES[$i]}"
        total_tokens="${CASE_TOTAL_TOKENS[$i]}"
        key="$model|$datasize"

        tp="${HP_TP[$key]:-}"
        ep="${HP_EP[$key]:-}"
        mbs="${HP_MBS[$key]:-}"
        gbs="${HP_GBS[$key]:-}"
        lr="${HP_LR[$key]:-}"
        hs="${HP_HS[$key]:-}"

        if [ -z "$tp" ] || [ -z "$ep" ] || [ -z "$mbs" ] || [ -z "$gbs" ] || [ -z "$lr" ] || [ -z "$hs" ]; then
            echo "  SKIP missing hyperparameters for datasize=$datasize"
            append_manifest_row "$model" "$case_tag" "$datasize" "" "" "" "" "" "missing_hyperparameters" "" ""
            skipped_missing=$((skipped_missing + 1))
            continue
        fi

        min_lr="$(min_lr_for "$lr")"
        tag="${case_tag}-${model_tag}-ds${datasize}B-ep${ep}-tp${tp}-mbs${mbs}-gbs${gbs}-hs${hs}"
        exp_name="${model_tag}-efficient-data-ablation-16n-4096sl-${gbs}gbsz-${tag}"
        exp_dir="$LOGS_BASE/$PROJECT_NAME/$exp_name"
        ckpt_dir="$exp_dir/checkpoints"
        tracker_file="$ckpt_dir/latest_checkpointed_iteration.txt"
        target_steps=$((total_tokens / (gbs * SEQ_LEN)))
        warmup_steps=$(( (target_steps * 3 + 99) / 100 ))
        final_results_file="$exp_dir/validation/final_checkpoint_eval_results.json"
        generic_results_file="$exp_dir/validation/eval_results.json"

        if [ ! -d "$exp_dir" ]; then
            echo "  SKIP missing run: $exp_name"
            append_manifest_row "$model" "$case_tag" "$datasize" "$exp_name" "$exp_dir" "$ckpt_dir" "$target_steps" "" "missing_run" "$final_results_file" ""
            skipped_missing=$((skipped_missing + 1))
            continue
        fi

        if [ ! -f "$tracker_file" ]; then
            echo "  SKIP no checkpoint tracker: $exp_name"
            append_manifest_row "$model" "$case_tag" "$datasize" "$exp_name" "$exp_dir" "$ckpt_dir" "$target_steps" "" "missing_checkpoint_tracker" "$final_results_file" ""
            skipped_missing=$((skipped_missing + 1))
            continue
        fi

        last_ckpt="$(tr -d '[:space:]' < "$tracker_file")"
        if [[ ! "$last_ckpt" =~ ^[0-9]+$ ]]; then
            echo "  SKIP invalid checkpoint tracker: $exp_name ($last_ckpt)"
            append_manifest_row "$model" "$case_tag" "$datasize" "$exp_name" "$exp_dir" "$ckpt_dir" "$target_steps" "$last_ckpt" "invalid_checkpoint_tracker" "$final_results_file" ""
            skipped_missing=$((skipped_missing + 1))
            continue
        fi

        if [ "$last_ckpt" -lt "$target_steps" ]; then
            echo "  SKIP incomplete: $exp_name (${last_ckpt}/${target_steps})"
            append_manifest_row "$model" "$case_tag" "$datasize" "$exp_name" "$exp_dir" "$ckpt_dir" "$target_steps" "$last_ckpt" "incomplete" "$final_results_file" ""
            skipped_incomplete=$((skipped_incomplete + 1))
            continue
        fi

        if json_has_result_for_iteration "$final_results_file" "$ckpt_dir" "$last_ckpt" || \
           json_has_result_for_iteration "$generic_results_file" "$ckpt_dir" "$last_ckpt"; then
            echo "  SKIP already validated: $exp_name (iter $last_ckpt)"
            append_manifest_row "$model" "$case_tag" "$datasize" "$exp_name" "$exp_dir" "$ckpt_dir" "$target_steps" "$last_ckpt" "already_validated" "$final_results_file" ""
            skipped_validated=$((skipped_validated + 1))
            continue
        fi

        if [ -z "$batch_script" ]; then
            batch_script="$(mktemp "/tmp/${model_tag}_final_validation_XXXXXX.sh")"
            chmod +x "$batch_script"
            {
                printf '#!/bin/bash\n'
                printf 'set -euo pipefail\n\n'
                printf 'echo "[%s] Starting batched final validations for %s"\n' '$(date)' "$(shell_quote "$model")"
                printf '\n'
            } > "$batch_script"
        fi

        append_batch_run "$batch_script" "$exp_name" "$exp_dir" "$ckpt_dir" "$mbs" "$tp" "$ep" "$hs" "$lr" "$min_lr" "$warmup_steps" "$final_results_file"
        echo "  QUEUED final validation: $exp_name (iter $last_ckpt)"
        append_manifest_row "$model" "$case_tag" "$datasize" "$exp_name" "$exp_dir" "$ckpt_dir" "$target_steps" "$last_ckpt" "queued_for_batch_submission" "$final_results_file" ""
        pending_count=$((pending_count + 1))
    done

    if [ "$pending_count" -gt 0 ]; then
        batch_job_id="$(submit_validation_batch "$model" "$model_tag" "$batch_script")"
        if [ "$DRY_RUN" = true ]; then
            echo "  READY batch validation job for $model with $pending_count runs"
        else
            echo "  SUBMITTED $batch_job_id: batch validation job for $model with $pending_count runs"
        fi
        submitted=$((submitted + pending_count))
    else
        echo "  No pending final validations for $model"
    fi

    echo ""
done

echo "========================================"
echo "submitted=$submitted"
echo "skipped_already_validated=$skipped_validated"
echo "skipped_incomplete=$skipped_incomplete"
echo "skipped_missing_or_invalid=$skipped_missing"
echo "manifest=$MANIFEST_PATH"
if [ "$DRY_RUN" = true ]; then
    echo "(DRY RUN - nothing submitted)"
fi
echo "========================================"
