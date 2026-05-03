#!/usr/bin/env python3
"""
One-shot migration: rename legacy SVM-prefixed artifacts to the new
DECODE-prefixed naming.

Two things change on disk:

  1. CSV column ``SVM_acc`` is renamed to ``decode_acc`` in every
     decoding-result CSV under ``derivatives/source_estimation/``.
     Files without an ``SVM_acc`` column are left alone.  The script
     skips any CSV that already has a ``decode_acc`` column (so reruns
     are safe).

  2. The output root directory is renamed:
       SVM_source              →  DECODE_source_space
       SVM_source_timeseries   →  DECODE_source_space_timeseries
     (Only renamed if the legacy directory exists.)

Default mode is ``--dry-run`` (the safe one): the script just lists
every change it would make.  Pass ``--apply`` to actually do it.

Usage:
    python migrate_svm_to_decode.py                # dry-run (default)
    python migrate_svm_to_decode.py --apply        # commit changes
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROJECT_ROOT


DERIVATIVES = PROJECT_ROOT / 'derivatives' / 'source_estimation'

DIR_RENAMES = [
    ('SVM_source',            'DECODE_source_space'),
    ('SVM_source_timeseries', 'DECODE_source_space_timeseries'),
]


def find_svm_acc_csvs(root: Path):
    """Yield (path, action) pairs for every CSV under root.

    action is one of:
      - 'rename-column'   → has SVM_acc, no decode_acc
      - 'already-renamed' → already has decode_acc (skip)
      - 'no-svm-acc'      → neither column present (skip)
      - 'unreadable'      → CSV failed to parse (skip with warning)
    """
    if not root.exists():
        return
    for path in root.rglob('*.csv'):
        try:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception as e:
            yield path, 'unreadable', str(e)
            continue
        has_svm = 'SVM_acc' in cols
        has_decode = 'decode_acc' in cols
        if has_svm and not has_decode:
            yield path, 'rename-column', None
        elif has_decode:
            yield path, 'already-renamed', None
        else:
            yield path, 'no-svm-acc', None


def rename_column_in_csv(path: Path):
    """Rewrite the CSV with the column renamed in place."""
    df = pd.read_csv(path)
    df = df.rename(columns={'SVM_acc': 'decode_acc'})
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Migrate SVM_acc → decode_acc and SVM_source → DECODE_source_space',
    )
    parser.add_argument('--apply', action='store_true',
                        help='Actually perform the renames (default is dry-run).')
    args = parser.parse_args()

    mode = 'APPLY' if args.apply else 'DRY-RUN'
    print(f'[{mode}] Source root: {DERIVATIVES}')
    if not DERIVATIVES.exists():
        print(f'  Derivatives root does not exist — nothing to do.')
        return

    # ── 1. CSV column renames ────────────────────────────────────────
    print('\n=== Step 1: CSV column SVM_acc → decode_acc ===')
    to_rename = []
    skipped_existing = 0
    skipped_noop = 0
    unreadable = []

    for path, action, err in find_svm_acc_csvs(DERIVATIVES):
        if action == 'rename-column':
            to_rename.append(path)
        elif action == 'already-renamed':
            skipped_existing += 1
        elif action == 'no-svm-acc':
            skipped_noop += 1
        elif action == 'unreadable':
            unreadable.append((path, err))

    print(f'  Scanned {len(to_rename) + skipped_existing + skipped_noop + len(unreadable)} CSV(s)')
    print(f'    {len(to_rename):>5} need column rename')
    print(f'    {skipped_existing:>5} already have decode_acc (skip)')
    print(f'    {skipped_noop:>5} have neither column (skip)')
    if unreadable:
        print(f'    {len(unreadable):>5} unreadable:')
        for p, e in unreadable:
            print(f'      {p}: {e}')

    if to_rename:
        print('\n  Files to rename column in:')
        for p in to_rename[:10]:
            print(f'    {p.relative_to(DERIVATIVES)}')
        if len(to_rename) > 10:
            print(f'    ... and {len(to_rename) - 10} more')

    if args.apply and to_rename:
        print(f'\n  Applying column rename to {len(to_rename)} file(s)...')
        for i, path in enumerate(to_rename, 1):
            try:
                rename_column_in_csv(path)
                if i % 50 == 0 or i == len(to_rename):
                    print(f'    {i}/{len(to_rename)} done')
            except Exception as e:
                print(f'    FAILED {path}: {e}')
        print('  Column rename complete.')

    # ── 2. Directory renames ─────────────────────────────────────────
    print('\n=== Step 2: Directory renames ===')
    for old, new in DIR_RENAMES:
        old_path = DERIVATIVES / old
        new_path = DERIVATIVES / new
        if not old_path.exists():
            print(f'  {old}/ does not exist — nothing to do.')
            continue
        if new_path.exists():
            print(f'  WARNING: both {old}/ and {new}/ exist. Skipping rename '
                  f'(merge manually if you want them combined).')
            continue
        print(f'  {old}/ → {new}/')
        if args.apply:
            old_path.rename(new_path)
            print(f'    Renamed.')

    # ── Summary ──────────────────────────────────────────────────────
    print()
    if args.apply:
        print(f'[{mode}] Done.')
    else:
        print(f'[{mode}] No changes made.  Re-run with --apply to commit.')


if __name__ == '__main__':
    main()
