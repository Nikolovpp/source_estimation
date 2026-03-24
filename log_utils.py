"""
Tee-style logging: duplicates stdout/stderr to a timestamped log file.

Usage (in runner scripts):
    from log_utils import setup_logging
    log_path = setup_logging(task, stim_class, method, atlas, feature_mode)
"""
import os
import sys
from datetime import datetime
from pathlib import Path

from config import SVM_OUTPUT_ROOT


class TeeStream:
    """Write to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, data):
        self.original.write(data)
        self.original.flush()
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def fileno(self):
        return self.original.fileno()

    def isatty(self):
        return self.original.isatty()


def setup_logging(task, stim_class, method, atlas, feature_mode,
                  runner_name='run'):
    """
    Redirect stdout and stderr to both the terminal and a log file.

    Parameters
    ----------
    task, stim_class, method, atlas, feature_mode : str
        Pipeline parameters used to build a descriptive log filename.
    runner_name : str
        Short label for the runner script (e.g. 'sequential', 'parallel',
        'parallel_lowram').

    Returns
    -------
    log_path : Path
        Absolute path to the log file that was created.
    """
    log_dir = SVM_OUTPUT_ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = (
        f'{timestamp}_{runner_name}_{task}_{stim_class}_{method}'
        f'_{atlas}_{feature_mode}.log'
    )
    log_path = log_dir / log_name

    log_file = open(log_path, 'w')

    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    print(f'Logging to: {log_path}')
    return log_path
