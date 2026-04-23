"""
Evaluate a trained checkpoint on per-modality validation sets and log results.

This script reuses Megatron's existing infrastructure (model loading, forward
step, distributed eval) but runs evaluation on each validation dataset
independently to produce per-modality loss numbers.

It logs results to:
  1. The same W&B run (via --wandb-save-dir pointing to the training run's logging dir)
  2. A JSON results file

Usage (launched via srun, same as training):
    python3 evaluate_validation.py \
        <all the same model/parallel args as training> \
        --load <checkpoint_dir> \
        --skip-train \
        --val-sets-dir /path/to/validation \
        --val-results-file /path/to/results.json
"""

import json
import math
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import List, Optional

import torch

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine, RerunMode
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.global_vars import get_wandb_writer
from megatron.training.initialize import initialize_megatron
from megatron.training.training import get_model, setup_model_and_optimizer
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    is_last_rank,
)
from megatron.training.datasets.data_samplers import build_pretraining_data_loader
from megatron.training.tokenizer.tokenizer_omni_metadata import populate_omni_metadata_from_tokenizer
from model_provider import model_provider

from pretrain_gpt import (
    forward_step,
    is_dataset_built_on_rank,
    loss_func,
)

# ── Validation set registry ──────────────────────────────────────────────────
VALIDATION_SETS = {
    "text_only": "text_only",
    "text_to_vision": "text_to_vision",
    "vision_to_text": "vision_to_text",
    "audio_text_interleaved": "audio_text_interleaved",
    "text_audio_interleaved": "text_audio_interleaved",
    "audio_only": "audio_only",
}


def add_validation_args(parser):
    """Add evaluation-specific arguments."""
    group = parser.add_argument_group(title='validation evaluation')
    group.add_argument(
        '--val-sets-dir',
        type=str,
        default=None,
        help='Directory containing per-modality validation set subdirectories '
             '(e.g., text_only/, vision_to_text/, etc.)',
    )
    group.add_argument(
        '--val-results-file',
        type=str,
        default=None,
        help='Path to write JSON results file',
    )
    group.add_argument(
        '--val-max-iters',
        type=int,
        default=None,
        help='Maximum number of eval iterations per validation set. '
             'If None, evaluate on all available data.',
    )
    group.add_argument(
        '--val-sets',
        nargs='+',
        default=None,
        help='Specific validation sets to evaluate (default: all found in val-sets-dir)',
    )
    group.add_argument(
        '--val-all-checkpoints',
        action='store_true',
        default=False,
        help='Evaluate ALL checkpoints in the --load directory (iter_XXXXXXX), '
             'not just the latest. Results are appended to the JSON file.',
    )
    group.add_argument(
        '--val-job-index',
        type=int,
        default=0,
        help='Job index for parallel evaluation (0-indexed). '
             'Used with --val-num-jobs to split checkpoints across SLURM jobs.',
    )
    group.add_argument(
        '--val-num-jobs',
        type=int,
        default=1,
        help='Total number of parallel evaluation jobs. '
             'Checkpoints are split round-robin across jobs.',
    )
    group.add_argument(
        '--val-save-every',
        type=int,
        default=10,
        help='Save results to JSON every N checkpoints (reduces I/O on large result files).',
    )
    group.add_argument(
        '--val-fraction',
        type=float,
        default=None,
        help='Fraction of checkpoints to evaluate (e.g. 0.33 for 1/3). '
             'Selects evenly spaced checkpoints from ALL available iterations '
             '(not just remaining), always including the last one. '
             'Already-evaluated ones in the selection are skipped.',
    )
    return parser


def get_val_bin_prefixes(val_dir: str) -> List[str]:
    """Walk a validation directory and return .bin file prefixes (without extension)."""
    prefixes = []
    for dp, _, fn in os.walk(val_dir, followlinks=True):
        for f in fn:
            full = os.path.join(dp, f)
            if full.lower().endswith(".bin"):
                prefix = full[:-4]  # strip .bin
                # Verify companion .idx exists
                if os.path.exists(prefix + ".idx"):
                    prefixes.append(prefix)
    return sorted(prefixes)


def build_val_dataset_config(args, prefixes):
    """Build a GPTDatasetConfig for a single validation set.

    Uses blend_per_split to put all data into the validation split only,
    with no weights (letting the builder use natural dataset sizes).
    """
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    populate_omni_metadata_from_tokenizer(args, tokenizer)

    modality_weights = {}
    omnimodal_config = getattr(args, "omnimodal_config", None)
    if omnimodal_config is not None:
        for modality in omnimodal_config.get("modalities", []):
            name = modality.get("name")
            if not name:
                continue
            weight = getattr(args, f"{name}_weight", None)
            if weight is not None:
                modality_weights[name] = weight
    for name in ("vision", "audio"):
        weight = getattr(args, f"{name}_weight", None)
        if weight is not None:
            modality_weights.setdefault(name, weight)

    # Use blend_per_split: [train=None, valid=blend, test=None]
    # No weights — builder will use natural dataset sizes (case 3d/2b)
    blend_per_split = [
        None,                   # train: nothing
        (prefixes, None),       # valid: all prefixes, no weights
        None,                   # test: nothing
    ]

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=None,
        blend_per_split=blend_per_split,
        split=None,  # Not used when blend_per_split is provided
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        modality_weights=modality_weights,
        loss_mask_token_ids=getattr(args, "loss_mask_token_ids", None),
        goldfish_loss=getattr(args, "goldfish_loss", False),
    )


def build_val_data_iterator(args, config, prefixes):
    """Build a data iterator for a single validation set."""
    # sizes: [train, valid, test] — None means "use all available data"
    num_samples = [0, None, 0]

    datasets = BlendedMegatronDatasetBuilder(
        GPTDataset,
        num_samples,
        is_dataset_built_on_rank,
        config,
    ).build()

    # datasets = [train_ds, valid_ds, test_ds]
    valid_ds = datasets[1]

    if valid_ds is None:
        return None, 0

    total_samples = len(valid_ds)

    # Build data loader
    valid_dataloader = build_pretraining_data_loader(valid_ds, 0)

    if valid_dataloader is None:
        return None, 0

    # Create cyclic iterator
    def cyclic_iter(dl):
        while True:
            for x in dl:
                yield x

    return iter(cyclic_iter(valid_dataloader)), total_samples


def evaluate_single(forward_step_func, data_iterator, model, eval_iters, args):
    """
    Run evaluation for a fixed number of iterations.
    Adapted from megatron.training.training.evaluate().
    Returns dict of {metric_name: value}.
    """
    # Turn on evaluation mode
    for model_module in model:
        model_module.eval()

    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)

    total_loss_dict = {}
    forward_backward_func = get_forward_backward_func()

    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // (
        args.micro_batch_size * args.data_parallel_size
    )

    with torch.no_grad():
        for iteration in range(1, eval_iters + 1):
            if iteration % 10 == 0:
                print_rank_0(f'  Eval iter {iteration}/{eval_iters}')

            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                forward_only=True,
            )

            # NOTE: removed torch.cuda.empty_cache() — it forces a CUDA sync
            # every iteration and severely degrades throughput during eval.

            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                for key in loss_dicts[0].keys():
                    if key not in total_loss_dict:
                        total_loss_dict[key] = torch.tensor(
                            [0.0, 0.0], dtype=torch.float
                        ).cuda()
                    val = [x[key].view(-1) for x in loss_dicts]

                    if val[0].numel() == 2:
                        val = torch.vstack(val).sum(dim=0)
                        torch.distributed.all_reduce(
                            val,
                            group=parallel_state.get_data_parallel_group(
                                with_context_parallel=True
                            ),
                        )
                        total_loss_dict[key] += val
                    elif val[0].numel() == 1:
                        val = torch.cat(val).sum()
                        total_loss_dict[key][0] += val
                        total_loss_dict[key][1] += len(loss_dicts)
                    else:
                        raise ValueError(
                            f"Invalid value shape: {val[0].shape} for key {key}"
                        )

    # Back to train mode
    for model_module in model:
        model_module.train()

    # Compute averages
    results = {}
    for key in total_loss_dict:
        numerator, denominator = total_loss_dict[key]
        if denominator > 0:
            results[key] = (numerator / denominator).item()
        else:
            results[key] = 0.0

    rerun_state_machine.set_mode(rerun_mode)
    return results


def discover_checkpoint_iterations(ckpt_dir):
    """Return sorted list of all checkpoint iterations in a directory."""
    iterations = []
    if not os.path.isdir(ckpt_dir):
        return iterations
    for entry in os.listdir(ckpt_dir):
        if entry.startswith("iter_") and os.path.isdir(os.path.join(ckpt_dir, entry)):
            try:
                iterations.append(int(entry.split("_")[1]))
            except (IndexError, ValueError):
                continue
    iterations.sort()
    return iterations


def build_precomputed_val_iterators(args, available_sets, val_max_iters):
    """Build dataset iterators for all validation sets once upfront.

    Returns a dict of {set_name: (data_iterator, total_samples, eval_iters, num_files)}
    for sets that have data, or {set_name: {"error": ...}} for failures.
    """
    precomputed = {}
    for set_name, (val_path, prefixes) in available_sets.items():
        print_rank_0(f"  Building dataset: {set_name} ({len(prefixes)} files)")
        try:
            config = build_val_dataset_config(args, prefixes)
            data_iterator, total_samples = build_val_data_iterator(args, config, prefixes)
        except Exception as e:
            tb = traceback.format_exc()
            print_rank_0(f"  ERROR building dataset for {set_name}: {e}\n{tb}")
            precomputed[set_name] = {"error": str(e) or repr(e)}
            continue

        total_samples_tensor = torch.tensor([total_samples], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(
            total_samples_tensor,
            src=parallel_state.get_tensor_model_parallel_src_rank(),
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        total_samples = total_samples_tensor.item()

        if total_samples == 0:
            print_rank_0(f"  WARNING: No data for {set_name}")
            precomputed[set_name] = {"error": "no data"}
            continue

        samples_per_iter = args.global_batch_size
        max_iters_from_data = max(1, total_samples // samples_per_iter)
        if val_max_iters is not None:
            eval_iters = min(val_max_iters, max_iters_from_data)
        else:
            eval_iters = max_iters_from_data

        precomputed[set_name] = (data_iterator, total_samples, eval_iters, len(prefixes))
        print_rank_0(f"    -> {total_samples} samples, {eval_iters} eval iters")

    return precomputed


def evaluate_checkpoint(model, iteration, args, precomputed_iterators, val_results_file):
    """Evaluate a single checkpoint using precomputed data iterators."""
    wandb_writer = get_wandb_writer()

    all_results = {
        "checkpoint_iteration": iteration,
        "checkpoint_path": args.load,
        "learning_rate": args.lr,
        "min_learning_rate": args.min_lr,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "seed": args.seed,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_experts": args.num_experts,
        "seq_length": args.seq_length,
        "validation_sets": {},
    }

    for set_name, val_info in precomputed_iterators.items():
        if isinstance(val_info, dict) and "error" in val_info:
            all_results["validation_sets"][set_name] = val_info
            continue

        data_iterator, total_samples, eval_iters, num_files = val_info

        print_rank_0(f"\n{'='*60}")
        print_rank_0(f"Evaluating: {set_name} (samples={total_samples}, iters={eval_iters})")
        print_rank_0(f"{'='*60}")

        t0 = time.time()
        results = evaluate_single(forward_step, data_iterator, model, eval_iters, args)
        elapsed = time.time() - t0

        set_results = {
            "num_files": num_files,
            "eval_iters": eval_iters,
            "eval_time_seconds": round(elapsed, 2),
        }
        for key, value in results.items():
            set_results[key] = value
            ppl = math.exp(min(20, value))
            set_results[f"{key}_ppl"] = ppl

        all_results["validation_sets"][set_name] = set_results

        print_rank_0(f"\n  Results for {set_name}:")
        for key, value in results.items():
            ppl = math.exp(min(20, value))
            print_rank_0(f"    {key}: {value:.6f} (PPL: {ppl:.4f})")

        if wandb_writer and is_last_rank():
            wandb_metrics = {}
            for key, value in results.items():
                wandb_metrics[f"val/{set_name}/{key}"] = value
                wandb_metrics[f"val/{set_name}/{key}_ppl"] = math.exp(min(20, value))
            wandb_writer.log(wandb_metrics, step=iteration)

    # Print summary
    print_rank_0(f"\n{'='*60}")
    print_rank_0("VALIDATION SUMMARY")
    print_rank_0(f"{'='*60}")
    print_rank_0(f"Checkpoint iteration: {iteration}")
    for set_name, set_results in all_results["validation_sets"].items():
        if "error" in set_results:
            print_rank_0(f"  {set_name}: ERROR - {set_results['error']}")
        elif "lm loss" in set_results:
            loss = set_results["lm loss"]
            ppl = set_results.get("lm loss_ppl", math.exp(min(20, loss)))
            print_rank_0(f"  {set_name}: loss={loss:.6f}, PPL={ppl:.4f}")
        else:
            print_rank_0(f"  {set_name}: {set_results}")

    return all_results


def save_results_buffered(results_buffer, val_results_file, args):
    """Save a batch of checkpoint results to the JSON file at once."""
    if not results_buffer:
        return
    if is_last_rank():
        os.makedirs(os.path.dirname(os.path.abspath(val_results_file)), exist_ok=True)
        existing_results = {}
        if os.path.isfile(val_results_file):
            try:
                with open(val_results_file, 'r') as f:
                    existing_results = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_results = {}
        for result_key, all_results in results_buffer:
            existing_results[result_key] = all_results
        with open(val_results_file, 'w') as f:
            json.dump(existing_results, f, indent=2)
        print_rank_0(f"\nSaved {len(results_buffer)} checkpoint results to: {val_results_file}")


def main():
    # Initialize Megatron with our extra args
    initialize_megatron(
        extra_args_provider=add_validation_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer', 'skip_train': True},
    )

    args = get_args()
    val_sets_dir = args.val_sets_dir
    val_results_file = args.val_results_file
    val_max_iters = args.val_max_iters
    val_set_names = args.val_sets
    val_all_checkpoints = args.val_all_checkpoints

    if val_sets_dir is None:
        print_rank_0("ERROR: --val-sets-dir is required")
        sys.exit(1)

    if val_results_file is None:
        val_results_file = os.path.join(val_sets_dir, "eval_results.json")

    # Setup model and load checkpoint
    print_rank_0("Setting up model and loading checkpoint...")
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
    )

    iteration = args.iteration
    print_rank_0(f"Loaded checkpoint at iteration {iteration}")

    if iteration == 0 and not val_all_checkpoints:
        print_rank_0("WARNING: Checkpoint iteration is 0 — no checkpoint was loaded!")
        print_rank_0("         Check that --load points to a valid checkpoint directory")
        print_rank_0("         with a latest_checkpointed_iteration.txt file.")

    # Discover validation sets
    available_sets = {}
    for name, subdir in VALIDATION_SETS.items():
        val_path = os.path.join(val_sets_dir, subdir)
        if os.path.isdir(val_path):
            prefixes = get_val_bin_prefixes(val_path)
            if prefixes:
                available_sets[name] = (val_path, prefixes)
                print_rank_0(f"  Found validation set: {name} ({len(prefixes)} files)")
            else:
                print_rank_0(f"  Skipping {name}: no valid .bin/.idx pairs found")
        else:
            print_rank_0(f"  Skipping {name}: directory not found ({val_path})")

    # Filter to requested sets
    if val_set_names:
        available_sets = {k: v for k, v in available_sets.items() if k in val_set_names}

    if not available_sets:
        print_rank_0("ERROR: No validation sets found!")
        sys.exit(1)

    print_rank_0(f"\nWill evaluate on {len(available_sets)} validation sets")

    # Pre-build all dataset iterators once (reused across checkpoints)
    print_rank_0("\nBuilding validation dataset iterators (one-time cost)...")
    precomputed = build_precomputed_val_iterators(args, available_sets, val_max_iters)
    print_rank_0("Dataset iterators ready.\n")

    val_job_index = args.val_job_index
    val_num_jobs = args.val_num_jobs
    val_save_every = args.val_save_every
    val_fraction = args.val_fraction

    if val_all_checkpoints:
        # Discover all checkpoint iterations and evaluate each one
        all_iters = discover_checkpoint_iterations(args.load)
        if not all_iters:
            print_rank_0(f"ERROR: No checkpoint iterations found in {args.load}")
            sys.exit(1)

        # Skip iterations that already have results
        existing_results = {}
        if os.path.isfile(val_results_file):
            try:
                with open(val_results_file, 'r') as f:
                    existing_results = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_results = {}
        done_iters = set()
        for key in existing_results:
            if f"{args.load}::iter" in key:
                try:
                    done_iters.add(int(key.split("::iter")[1]))
                except (IndexError, ValueError):
                    pass
        # Select evenly spaced subset if --val-fraction is set
        # This selects from ALL checkpoints (not just remaining) to get
        # a consistent spacing, then filters out already-evaluated ones.
        if val_fraction is not None and 0 < val_fraction < 1:
            n_total = len(all_iters)
            n_select = max(1, int(round(n_total * val_fraction)))
            # Evenly spaced indices, always including the last checkpoint
            if n_select >= n_total:
                selected_iters = set(all_iters)
            else:
                indices = [int(round(i * (n_total - 1) / (n_select - 1))) for i in range(n_select)] if n_select > 1 else [n_total - 1]
                selected_iters = set(all_iters[i] for i in indices)
                selected_iters.add(all_iters[-1])  # always include last
            remaining_iters = [it for it in all_iters if it in selected_iters and it not in done_iters]
        else:
            remaining_iters = [it for it in all_iters if it not in done_iters]

        # Split remaining checkpoints across parallel jobs (round-robin)
        if val_num_jobs > 1:
            remaining_iters = [it for i, it in enumerate(remaining_iters) if i % val_num_jobs == val_job_index]

        print_rank_0(f"\n{'='*60}")
        print_rank_0(f"ALL-CHECKPOINTS MODE")
        print_rank_0(f"  Total checkpoints found: {len(all_iters)}")
        if val_fraction is not None:
            print_rank_0(f"  Fraction: {val_fraction} — selected {len(selected_iters)} evenly spaced (always incl. last)")
        print_rank_0(f"  Already evaluated: {len(done_iters)}")
        if val_num_jobs > 1:
            print_rank_0(f"  This job: {val_job_index+1}/{val_num_jobs} — assigned {len(remaining_iters)} checkpoints")
        else:
            print_rank_0(f"  Remaining to evaluate: {len(remaining_iters)}")
        print_rank_0(f"{'='*60}")

        if not remaining_iters:
            print_rank_0("All checkpoints already evaluated (or none assigned). Nothing to do.")
            return

        results_buffer = []
        for idx, ckpt_iter in enumerate(remaining_iters):
            print_rank_0(f"\n{'#'*60}")
            print_rank_0(f"  Checkpoint {idx+1}/{len(remaining_iters)}: iteration {ckpt_iter}")
            print_rank_0(f"{'#'*60}")

            # Set the target iteration and reload weights
            args.ckpt_step = ckpt_iter
            args.iteration, args.num_floating_point_operations_so_far, args.tokens_so_far = \
                load_checkpoint(model, optimizer, opt_param_scheduler, strict=False)
            print_rank_0(f"  Loaded checkpoint at iteration {args.iteration}")

            all_results = evaluate_checkpoint(model, args.iteration, args, precomputed, val_results_file)
            result_key = f"{args.load}::iter{args.iteration}"
            results_buffer.append((result_key, all_results))

            # Flush buffer periodically
            if len(results_buffer) >= val_save_every:
                save_results_buffered(results_buffer, val_results_file, args)
                results_buffer = []

        # Flush remaining results
        save_results_buffered(results_buffer, val_results_file, args)

        print_rank_0(f"\n{'='*60}")
        print_rank_0(f"ALL-CHECKPOINTS COMPLETE: evaluated {len(remaining_iters)} checkpoints")
        print_rank_0(f"Results in: {val_results_file}")
        print_rank_0(f"{'='*60}")
    else:
        # Single checkpoint mode (original behavior)
        all_results = evaluate_checkpoint(model, iteration, args, precomputed, val_results_file)
        result_key = f"{args.load}::iter{iteration}"
        save_results_buffered([(result_key, all_results)], val_results_file, args)

    # Finish W&B
    wandb_writer = get_wandb_writer()
    if wandb_writer and is_last_rank():
        wandb_writer.finish()


if __name__ == "__main__":
    main()
