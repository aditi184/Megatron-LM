#!/bin/bash
#
# Submit the requested data-ablation experiment set using hyperparamters.csv.
#
# The CSV is expected to contain tab-separated rows:
#   model, tp, ep, mbs, gbs, lr, datasize, hs
#
# Hyperparameters are selected by (model, datasize):
#   - text+vision and text+audio use datasize=100
#   - vision-only and audio-only use datasize={5,10,15,20,25}
#   - vision/audio interference uses datasize=150
#
# Usage:
#   bash submit_requested_data_ablations.sh --dry-run
#   bash submit_requested_data_ablations.sh
#   bash submit_requested_data_ablations.sh --models "Model 2,Model 7"
#   bash submit_requested_data_ablations.sh --skip-validation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_SCRIPT="$SCRIPT_DIR/master_data_ablations.sh"
HYPERPARAMS_CSV="$SCRIPT_DIR/hyperparamters.csv"

DATA_DIR=/iopsstor/scratch/cscs/aditikhandelwal/pretraining-datasets
TEXT_PATH="$DATA_DIR/merged/text_only_merged"
VISION_I2T_PATH="$DATA_DIR/vision_text_interleaved/train_img_conditioned_on_text/llavaOv1_5_Midtrain_paired_apertus8b_emu3p5_merged"
VISION_T2I_PATH="$DATA_DIR/vision_text_interleaved/train_text_conditioned_on_img/llavaOv1_5_Midtrain_paired_apertus8b_emu3p5_merged"
AUDIO_ONLY_PATH="$DATA_DIR/audio_only/train/merged"
AUDIO_TEXT_PATH="$DATA_DIR/audio_text_interleaved/train/MERGED_0.1_36B/merged"

MODAL_TOKENS_B=(5 10 15 20 25)
INTERFERENCE_SPLITS=("45:5" "41.5:8.5" "37.5:12.5" "33:17" "25:25")

MODEL_FILTER=""
CHECKPOINT_STEPS=1000
SEED=28
AUTO_REQUEUE=false
SKIP_VALIDATION=false
DRY_RUN=false
WALLTIME=""
RESERVATION=""
TAG_PREFIX=""
TAG_SUFFIX=""

declare -A HP_TP HP_EP HP_MBS HP_GBS HP_LR HP_HS
MODELS=()

usage() {
    sed -n '1,17p' "$0"
}

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    echo "$s"
}

normalize_model() {
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
    CASE_GROUPS+=("$1")
    CASE_TAGS+=("$2")
    CASE_DATASIZES+=("$3")
    CASE_TOTAL_TOKENS+=("$4")
    CASE_DATA_PATHS+=("$5")
}

build_cases() {
    CASE_GROUPS=()
    CASE_TAGS=()
    CASE_DATASIZES=()
    CASE_TOTAL_TOKENS=()
    CASE_DATA_PATHS=()

    local vb ab split vis_b aud_b text_w vis_w aud_w data_path

    for vb in "${MODAL_TOKENS_B[@]}"; do
        text_w=$(weight "$(python3 -c "print(100.0 - float('$vb'))")" 100)
        vis_w=$(weight "$vb" 200)
        data_path="$text_w $TEXT_PATH $vis_w $VISION_I2T_PATH $vis_w $VISION_T2I_PATH"
        append_case "text+vision" "text-vision-${vb}Bvis" 100 100000000000 "$data_path"
    done

    for vb in "${MODAL_TOKENS_B[@]}"; do
        data_path="0.5 $VISION_I2T_PATH 0.5 $VISION_T2I_PATH"
        append_case "vision-only" "vision-only-${vb}B" "$vb" "$(to_tokens "$vb")" "$data_path"
    done

    for ab in "${MODAL_TOKENS_B[@]}"; do
        text_w=$(weight "$(python3 -c "print(100.0 - float('$ab'))")" 100)
        aud_w=$(weight "$ab" 200)
        data_path="$text_w $TEXT_PATH $aud_w $AUDIO_ONLY_PATH $aud_w $AUDIO_TEXT_PATH"
        append_case "text+audio" "text-audio-${ab}Baud" 100 100000000000 "$data_path"
    done

    for ab in "${MODAL_TOKENS_B[@]}"; do
        data_path="0.5 $AUDIO_ONLY_PATH 0.5 $AUDIO_TEXT_PATH"
        append_case "audio-only" "audio-only-${ab}B" "$ab" "$(to_tokens "$ab")" "$data_path"
    done

    for split in "${INTERFERENCE_SPLITS[@]}"; do
        vis_b="${split%%:*}"
        aud_b="${split##*:}"
        text_w=$(weight 100 150)
        vis_w=$(weight "$vis_b" 300)
        aud_w=$(weight "$aud_b" 300)
        data_path="$text_w $TEXT_PATH $vis_w $VISION_I2T_PATH $vis_w $VISION_T2I_PATH $aud_w $AUDIO_ONLY_PATH $aud_w $AUDIO_TEXT_PATH"
        append_case "interference" "interference-${vis_b}Bvis-${aud_b}Baud" 150 150000000000 "$data_path"
    done
}

validate_parallel_config() {
    local tp="$1"
    local ep="$2"
    local mbs="$3"
    local gbs="$4"
    local world_size=64
    local dp

    if [ $((world_size % tp)) -ne 0 ]; then
        echo "WARNING: Invalid TP=$tp: $world_size GPUs is not divisible by TP" >&2
        return
    fi

    dp=$((world_size / tp))
    if [ $((dp % ep)) -ne 0 ]; then
        echo "WARNING: Invalid EP=$ep with TP=$tp: DP=$dp is not divisible by EP" >&2
        return
    fi

    if [ $((gbs % (dp * mbs))) -ne 0 ]; then
        echo "WARNING: GBS=$gbs, MBS=$mbs, TP=$tp is not divisible by DP*MBS=$((dp * mbs))" >&2
    fi
}

submit_job() {
    local model="$1"
    local case_tag="$2"
    local datasize="$3"
    local total_tokens="$4"
    local data_path="$5"
    local key="$model|$datasize"
    local tp="${HP_TP[$key]:-}"
    local ep="${HP_EP[$key]:-}"
    local mbs="${HP_MBS[$key]:-}"
    local gbs="${HP_GBS[$key]:-}"
    local lr="${HP_LR[$key]:-}"
    local hs="${HP_HS[$key]:-}"
    local min_lr model_tag tag job_id

    if [ -z "$tp" ] || [ -z "$ep" ] || [ -z "$mbs" ] || [ -z "$gbs" ] || [ -z "$lr" ] || [ -z "$hs" ]; then
        echo "Missing hyperparameters for $model datasize=$datasize" >&2
        exit 1
    fi

    validate_parallel_config "$tp" "$ep" "$mbs" "$gbs"

    min_lr="$(min_lr_for "$lr")"
    model_tag="$(normalize_model "$model")"
    tag="${TAG_PREFIX}${case_tag}-${model_tag}-ds${datasize}B-ep${ep}-tp${tp}-mbs${mbs}-gbs${gbs}-hs${hs}${TAG_SUFFIX:+-$TAG_SUFFIX}"

    echo "  ${tag} | tokens=${total_tokens} | lr=${lr} min_lr=${min_lr}"

    if [ "$DRY_RUN" = false ]; then
        local -a sbatch_args cmd
        sbatch_args=(--parsable)
        if [ -n "$WALLTIME" ]; then
            sbatch_args+=(--time="$WALLTIME")
        fi
        if [ -n "$RESERVATION" ]; then
            sbatch_args+=(--reservation="$RESERVATION")
        fi

        cmd=(env
            "OVERRIDE_TOTAL_TOKENS=$total_tokens"
            "OVERRIDE_DATA_PATH=$data_path"
            "OVERRIDE_TAG=$tag"
            "OVERRIDE_MODEL_NAME=$model_tag"
            "OVERRIDE_LR=$lr"
            "OVERRIDE_MIN_LR=$min_lr"
            "OVERRIDE_TP=$tp"
            "OVERRIDE_EP=$ep"
            "OVERRIDE_MBS=$mbs"
            "OVERRIDE_GBS=$gbs"
            "OVERRIDE_HIDDEN_SIZE=$hs"
            "OVERRIDE_MOE_FFN_HIDDEN_SIZE=$hs"
            "OVERRIDE_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$hs"
            "OVERRIDE_CHECKPOINT_STEPS=$CHECKPOINT_STEPS"
            "OVERRIDE_SEED=$SEED"
            "OVERRIDE_AUTO_REQUEUE=$AUTO_REQUEUE"
            "OVERRIDE_SKIP_VALIDATION=$SKIP_VALIDATION"
            sbatch "${sbatch_args[@]}" "$MODEL_SCRIPT")

        job_id=$("${cmd[@]}")
        echo "    -> Submitted job $job_id"
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --models) MODEL_FILTER="$2"; shift 2 ;;
        --hyperparams-csv) HYPERPARAMS_CSV="$2"; shift 2 ;;
        --checkpoint-steps) CHECKPOINT_STEPS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --time) WALLTIME="$2"; shift 2 ;;
        --reservation) RESERVATION="$2"; shift 2 ;;
        --tag-prefix) TAG_PREFIX="$2"; shift 2 ;;
        --tag-suffix) TAG_SUFFIX="$2"; shift 2 ;;
        --auto-requeue) AUTO_REQUEUE=true; shift ;;
        --skip-validation) SKIP_VALIDATION=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ ! -f "$MODEL_SCRIPT" ]; then
    echo "Missing model script: $MODEL_SCRIPT" >&2
    exit 1
fi

load_hyperparams
build_cases

echo ""
echo "============================================================="
echo "  Requested data-ablation experiments"
echo "============================================================="
echo "  script=$MODEL_SCRIPT"
echo "  hyperparams=$HYPERPARAMS_CSV"
echo "  models=${MODEL_FILTER:-all from CSV}"
echo "  data cases=${#CASE_TAGS[@]}"
echo "  validation: skip=$SKIP_VALIDATION"
if [ "$DRY_RUN" = true ]; then
    echo "  dry-run=true"
fi
echo ""

total_jobs=0
for model in "${MODELS[@]}"; do
    if ! model_selected "$model"; then
        continue
    fi

    echo "Model: $model"
    for i in "${!CASE_TAGS[@]}"; do
        submit_job "$model" "${CASE_TAGS[$i]}" "${CASE_DATASIZES[$i]}" "${CASE_TOTAL_TOKENS[$i]}" "${CASE_DATA_PATHS[$i]}"
        total_jobs=$((total_jobs + 1))
    done
    echo ""
done

echo "========================================"
echo "Total jobs: $total_jobs"
if [ "$DRY_RUN" = true ]; then
    echo "(DRY RUN - nothing submitted)"
fi
echo "========================================"
