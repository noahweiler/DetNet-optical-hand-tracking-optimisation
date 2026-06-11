"""
benchmark_inference.py — Controlled latency benchmark for DetNet (pruned or baseline).

This is the rigorous counterpart to evaluate_detnet.py's eval-pass timing. It
removes every noise source we identified:

  - Fixed synthetic input (no dataloader, no cv2/PIL/affine in the loop).
  - torch.cuda.Event GPU-side timing (not perf_counter + synchronize, which
    folds CPU dispatch + sync overhead into the measured window).
  - cuDNN autotuner warmed up (cudnn.benchmark=True) so kernel selection is
    stable and fast before timing starts.
  - Explicit warm-up iterations discarded (covers CUDA JIT + autotune).
  - Optional GPU clock locking (printed instructions — locking needs admin).

It loads checkpoints the same two ways evaluate_detnet.py does (baseline
state_dict, or pruned + fine-tuned via --arch) by reusing its load_model().

Output: <out_dir>/<ckpt_stem>_<device>_bs<batch>/summary.json — a JSON whose
'latency' block uses the SAME keys (total_ms_per_frame, total_ms_std) as
evaluate_detnet.py, so plot_pruning_results.py can consume it for a clean Fig 7
(point it at this folder via --results_dir, or wire a --latency_dir flag).

Usage:
  python benchmark_inference.py --checkpoint new_check_point/ckp_detnet_71.pth
  python benchmark_inference.py --checkpoint checkpoints_finetuned/l1_25pct/ft_l1_25pct_rhdbest.pth \\
      --arch pruned_architectures/detnet_l1_25pct_ep71_noft.pth
  python benchmark_inference.py --checkpoint ... --device cpu --batch_size 1
  python benchmark_inference.py --checkpoint ... --batch_size 32 --iters 1000
  python benchmark_inference.py --checkpoint ... --lock_clock_hint
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows + cv2/numpy/torch: silence OMP Error #15

import sys
try:
    # Box-drawing chars in the report crash on Windows cp1252 stdout (e.g. when
    # piped/Tee-Object'd). Force UTF-8 so output is robust everywhere.
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse
import json
import subprocess
import time
from datetime import datetime

import numpy as np
import torch

# Reuse evaluate_detnet's checkpoint loading + model-stat helpers so the two
# scripts can never disagree about how a model is built or measured.
from evaluate_detnet import (
    load_model, count_params, compute_flops, get_memory_stats, INPUT_SIZE,
)


# ── GPU clock helpers ───────────────────────────────────────────────────────

def query_gpu_clock_mhz():
    """Best-effort current SM clock via nvidia-smi; None if unavailable."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.sm', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def print_clock_lock_hint(device):
    """Print the nvidia-smi commands to lock clocks for a reproducible run.
    We only PRINT them — locking needs an elevated shell and is the user's
    deliberate choice, not something a benchmark script should do silently."""
    print('\n── GPU clock-lock instructions (run an *Administrator* terminal) ──')
    if device.type != 'cuda':
        print('  Device is CPU — clock locking is not applicable.')
        return
    name = torch.cuda.get_device_name(device)
    clk  = query_gpu_clock_mhz()
    print(f'  GPU: {name}' + (f'   current SM clock: {clk} MHz' if clk else ''))
    print('  List supported clocks : nvidia-smi -q -d SUPPORTED_CLOCKS')
    print('  Lock SM clock         : nvidia-smi -lgc <MHz>,<MHz>   (pick a mid value, not max)')
    print('  Lock memory clock     : nvidia-smi -lmc <MHz>         (driver 460+)')
    print('  Reset when done       : nvidia-smi -rgc ; nvidia-smi -rmc')
    print('  Then re-run this script WITHOUT --lock_clock_hint to benchmark.\n')


# ── Timing core ─────────────────────────────────────────────────────────────

def benchmark_cuda(model, x, warmup, iters):
    """Per-iteration latency (ms) using CUDA events — no CPU dispatch/sync in
    the measured window. Returns a (iters,) numpy array of milliseconds."""
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(warmup):                 # JIT + cuDNN autotune
            model(x)
        torch.cuda.synchronize()

        times = np.empty(iters, dtype=np.float64)
        for i in range(iters):
            starter.record()
            model(x)
            ender.record()
            torch.cuda.synchronize()            # wait for THIS iter only
            times[i] = starter.elapsed_time(ender)   # ms, GPU-measured
    return times


def benchmark_cpu(model, x, warmup, iters):
    """Per-iteration latency (ms) using perf_counter (correct for CPU, where
    there is no async GPU queue to account for)."""
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = np.empty(iters, dtype=np.float64)
        for i in range(iters):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            times[i] = (t1 - t0) * 1000.0
    return times


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Controlled DetNet inference-latency benchmark')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='DetNet state_dict (.pth). Baseline: use alone. '
                             'Pruned + fine-tuned: also pass --arch.')
    parser.add_argument('--arch', type=str, default=None,
                        help='Matching pruned architecture (_noft.pth). Omit for the baseline.')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                        help='Where to benchmark (default: cuda; falls back to cpu if no GPU).')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Inference batch size (default: 1 = single-frame latency).')
    parser.add_argument('--warmup', type=int, default=50,
                        help='Warm-up iterations discarded before timing (default: 50).')
    parser.add_argument('--iters', type=int, default=500,
                        help='Timed iterations (default: 500).')
    parser.add_argument('--input_size', type=int, default=INPUT_SIZE,
                        help=f'Square input resolution (default: {INPUT_SIZE}).')
    parser.add_argument('--out_dir', type=str, default='benchmark_results',
                        help='Folder for the per-run summary.json (default: benchmark_results).')
    parser.add_argument('--lock_clock_hint', action='store_true',
                        help='Print nvidia-smi clock-lock commands and exit (no benchmark).')
    args = parser.parse_args()

    want_cuda = (args.device == 'cuda') and torch.cuda.is_available()
    device = torch.device('cuda:0' if want_cuda else 'cpu')
    if args.device == 'cuda' and not want_cuda:
        print('CUDA requested but unavailable — falling back to CPU.')

    if args.lock_clock_hint:
        print_clock_lock_hint(device)
        return

    print(f'Device : {device}')
    model = load_model(args.checkpoint, device, arch_path=args.arch)
    model.eval()

    # Stable, fast kernels for a fixed input shape.
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Fixed synthetic input — created ONCE, reused every iteration (no dataloader).
    torch.manual_seed(0)
    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size,
                    device=device, dtype=torch.float32)

    # Model-level stats (consistent with evaluate_detnet.py).
    n_params       = count_params(model)
    flops_info     = compute_flops(model, device)   # per single forward (batch 1)
    mem_info       = get_memory_stats(model, device)
    gpu_name       = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    sm_clock_start = query_gpu_clock_mhz() if device.type == 'cuda' else None

    print(f'Benchmarking: warmup={args.warmup}, iters={args.iters}, '
          f'batch={args.batch_size}, input={args.input_size}^2')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        times_ms = benchmark_cuda(model, x, args.warmup, args.iters)
        peak_gb  = torch.cuda.max_memory_allocated(device) / 1e9
        timing_method = 'cuda_event'
    else:
        times_ms = benchmark_cpu(model, x, args.warmup, args.iters)
        peak_gb  = mem_info['peak_gb']
        timing_method = 'perf_counter'

    sm_clock_end = query_gpu_clock_mhz() if device.type == 'cuda' else None

    # Per-batch stats, then per-frame (= per-batch / batch_size).
    ms_batch_mean = float(times_ms.mean())
    ms_batch_std  = float(times_ms.std())
    ms_per_frame  = ms_batch_mean / args.batch_size
    ms_frame_std  = ms_batch_std  / args.batch_size
    fps           = (args.batch_size * 1000.0 / ms_batch_mean) if ms_batch_mean > 0 else None

    gflops, gflops_note = flops_info

    # ── Report ──────────────────────────────────────────────────────────────
    W = 60
    print(f'\n{"═" * W}')
    print(f'  Controlled Inference Benchmark')
    print(f'{"─" * W}')
    print(f'  {"GPU/Device":<24}: {gpu_name}')
    if sm_clock_start is not None:
        drift = '' if sm_clock_end is None else f'  (end {sm_clock_end} MHz)'
        print(f'  {"SM clock at start":<24}: {sm_clock_start} MHz{drift}')
    print(f'  {"Checkpoint":<24}: {args.checkpoint}')
    if args.arch:
        print(f'  {"Architecture (_noft)":<24}: {args.arch}')
    print(f'  {"Parameters":<24}: {n_params/1e6:>9.3f} M')
    if gflops is not None:
        print(f'  {"GFLOPs (1 fwd)":<24}: {gflops:>9.3f}')
    print(f'  {"Batch size":<24}: {args.batch_size}')
    print(f'  {"Warm-up / timed":<24}: {args.warmup} / {args.iters}')
    print(f'  {"Timing method":<24}: {timing_method}')
    print(f'  {"─" * 50}')
    print(f'  {"Latency / batch":<24}: {ms_batch_mean:>8.4f} ± {ms_batch_std:<7.4f} ms')
    print(f'  {"Latency / frame":<24}: {ms_per_frame:>8.4f} ± {ms_frame_std:<7.4f} ms')
    print(f'  {"min / median / p95":<24}: '
          f'{times_ms.min():.4f} / {np.median(times_ms):.4f} / '
          f'{np.percentile(times_ms, 95):.4f} ms')
    print(f'  {"Throughput":<24}: {fps:>9.2f} fps')
    print(f'  {"Peak inference mem":<24}: {peak_gb:>9.4f} GB')
    print(f'{"═" * W}\n')

    # ── Persist (schema-compatible with evaluate_detnet.py's latency block) ──
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    run_name  = f'{ckpt_stem}_{device.type}_bs{args.batch_size}'
    run_dir   = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    summary = {
        'timestamp':   datetime.now().isoformat(timespec='seconds'),
        'checkpoint':  args.checkpoint,
        'arch':        args.arch,
        'device':      str(device),
        'gpu_name':    gpu_name,
        'batch_size':  args.batch_size,
        'input_size':  args.input_size,
        'warmup_iters': args.warmup,
        'timed_iters':  args.iters,
        'timing_method': timing_method,
        'sm_clock_mhz_start': sm_clock_start,
        'sm_clock_mhz_end':   sm_clock_end,
        'model': {
            'param_count':   int(n_params),
            'param_count_M': n_params / 1e6,
            'ram_gb':        mem_info['model_gb'],
            'total_mb':      mem_info['total_mb'],
        },
        'compute': {
            'gflops_forward': gflops,
            'gflops_note':    gflops_note,
        },
        'latency': {
            # Keys mirror evaluate_detnet.py so plot_pruning_results.py reads
            # this transparently for a clean, low-variance Fig 7.
            'total_ms_per_frame': ms_per_frame,
            'total_ms_std':       ms_frame_std,
            'ms_per_batch_mean':  ms_batch_mean,
            'ms_per_batch_std':   ms_batch_std,
            'ms_min':             float(times_ms.min()),
            'ms_median':          float(np.median(times_ms)),
            'ms_p95':             float(np.percentile(times_ms, 95)),
            'fps':                fps,
            'timing_method':      timing_method,
        },
        'memory': {
            'peak_inference_gb': peak_gb,
            'peak_source':       device.type.upper(),
        },
    }

    json_path = os.path.join(run_dir, 'summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved -> {run_dir}')


if __name__ == '__main__':
    main()
