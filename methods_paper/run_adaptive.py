#!/usr/bin/env python3
"""Auto-size a heavy GC run to the machine's FREE cores/RAM, leaving headroom.

Wraps ``gc_estimator_order_sweep.py``.  It measures what is *currently free*
(cores minus the 1-min load average; RAM from /proc/meminfo MemAvailable),
reserves a safety margin so the box never saturates or swaps, then picks:

  --reduce-jobs : Stage A (reduce+cache) is RAM-bound — each worker loads a
                  ~3-4 GB vertex file — so this is capped by free-RAM / per-worker.
  --gc-jobs     : Stage B (GC+TRGC sweep) is RAM-light (tiny reduced caches),
                  so this is capped only by free cores.

Everything else on the command line is forwarded verbatim to the sweep script.
If you pass your own --reduce-jobs / --gc-jobs they are respected (not overridden).
No psutil dependency (reads /proc); Linux only.

Examples
--------
    # let it size itself and run overtProd
    python methods_paper/run_adaptive.py -- --task overtProd  --stim-class prodDiff
    # just show the plan, don't launch
    python methods_paper/run_adaptive.py --dry-run -- --task perception --stim-class percDiff
    # be more conservative (leave 8 cores + 64 GB free), pin per-worker RAM
    python methods_paper/run_adaptive.py --core-reserve 8 --ram-reserve-gb 64 \
        --per-worker-gb 4 -- --task overtProd --stim-class prodDiff

A leading ``--`` before the sweep args is optional but recommended (keeps the
two arg sets unambiguous).
"""
import os, sys, math, argparse, subprocess, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, 'gc_estimator_order_sweep.py')


def read_meminfo_gb():
    """(MemTotal, MemAvailable) in GiB from /proc/meminfo."""
    total = avail = None
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    total = int(line.split()[1]) / 1024 / 1024
                elif line.startswith('MemAvailable:'):
                    avail = int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return total, avail


def n_cores():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def load1():
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return 0.0


def estimate_per_worker_gb(default=4.0):
    """Peak RAM for one Stage-A reduce worker, from a real vertex file size.

    The vertex npz barely compresses (~1.1x); a worker holds the decompressed
    float32 arrays plus a transient float64 copy of the largest ROI during the
    FIXPC1 SVD, so peak ~ disk_size x ~1.7.  Falls back to ``default`` if no
    cache is resolvable (e.g. config.env not set yet).
    """
    try:
        sys.path.insert(0, os.path.dirname(HERE))
        import config
        for task, stim in (('overtProd', 'prodDiff'), ('perception', 'percDiff')):
            for subj in config.SUBJECT_IDS:
                p = config.find_cached_npz(task, 'LCMV', 'custom',
                                           'vertex_selectkbest', True, subj, stim)
                if p and os.path.exists(p):
                    disk_gb = os.path.getsize(p) / 1024 ** 3
                    return max(3.0, disk_gb * 1.7)
    except Exception:
        pass
    return default


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--script', default=SWEEP, help='target sweep script')
    ap.add_argument('--per-worker-gb', type=float, default=0.0,
                    help='Stage-A RAM per worker (GiB); 0 = auto-estimate from a real file')
    ap.add_argument('--core-reserve', type=int, default=None,
                    help='cores to keep free (default = max(2, 15%% of cores))')
    ap.add_argument('--core-reserve-frac', type=float, default=0.15)
    ap.add_argument('--ram-reserve-gb', type=float, default=None,
                    help='RAM (GiB) to keep free (default = max(16, 12%% of total))')
    ap.add_argument('--ram-reserve-frac', type=float, default=0.12)
    ap.add_argument('--max-reduce-jobs', type=int, default=16,
                    help='hard cap on Stage-A workers — it is IO-bound (big vertex reads), '
                         'so more than this thrashes the disk without helping. Raise on fast '
                         'local NVMe.')
    ap.add_argument('--dry-run', action='store_true', help='print the plan, do not launch')
    args, passthrough = ap.parse_known_args()
    # drop a lone leading '--' separator if present
    if passthrough and passthrough[0] == '--':
        passthrough = passthrough[1:]

    cores = n_cores()
    ld = load1()
    core_reserve = (args.core_reserve if args.core_reserve is not None
                    else max(2, math.ceil(args.core_reserve_frac * cores)))
    free_cores = cores - ld
    usable = int(max(1, math.floor(free_cores) - core_reserve))

    mem_total, mem_avail = read_meminfo_gb()
    if mem_total is None:
        print('WARNING: /proc/meminfo unreadable — assuming 64 GiB total / 32 free')
        mem_total, mem_avail = 64.0, 32.0
    ram_reserve = (args.ram_reserve_gb if args.ram_reserve_gb is not None
                   else max(16.0, args.ram_reserve_frac * mem_total))
    ram_budget = max(0.0, mem_avail - ram_reserve)

    per_worker = args.per_worker_gb or estimate_per_worker_gb()
    reduce_by_ram = max(1, int(ram_budget // per_worker))
    reduce_jobs = max(1, min(usable, reduce_by_ram, args.max_reduce_jobs))
    gc_jobs = max(1, usable)

    # respect an explicit user choice
    user_reduce = any(a == '--reduce-jobs' or a.startswith('--reduce-jobs=') for a in passthrough)
    user_gc = any(a == '--gc-jobs' or a.startswith('--gc-jobs=') for a in passthrough)

    print('=' * 66)
    print('Adaptive launcher — resource snapshot')
    print(f'  cores (affinity)      : {cores}')
    print(f'  1-min load average    : {ld:.1f}  -> ~{free_cores:.1f} cores free')
    print(f'  core reserve (headroom): {core_reserve}  -> usable cores: {usable}')
    print(f'  RAM total / available  : {mem_total:.0f} / {mem_avail:.0f} GiB')
    print(f'  RAM reserve (headroom) : {ram_reserve:.0f} GiB -> budget: {ram_budget:.0f} GiB')
    print(f'  Stage-A RAM per worker : {per_worker:.1f} GiB'
          f'{" (auto)" if not args.per_worker_gb else ""}')
    print('-' * 66)
    print(f'  --reduce-jobs -> {reduce_jobs}'
          + f'  (caps: RAM {reduce_by_ram}, cores {usable}, IO {args.max_reduce_jobs})'
          + ('  [user override — not applied]' if user_reduce else ''))
    print(f'  --gc-jobs     -> {gc_jobs}'
          + ('  [user override — not applied]' if user_gc else ''))
    print('=' * 66)

    if usable < 2 or ram_budget < per_worker:
        print('WARNING: very little is free right now — the run will be tiny/serial.')

    cmd = [sys.executable, args.script, *passthrough]
    if not user_reduce:
        cmd += ['--reduce-jobs', str(reduce_jobs)]
    if not user_gc:
        cmd += ['--gc-jobs', str(gc_jobs)]
    print('launch:', ' '.join(cmd))
    if args.dry_run:
        print('(dry run — not launched)')
        return
    sys.stdout.flush()
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
