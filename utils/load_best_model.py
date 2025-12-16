import os
from pathlib2 import Path


def load_model_path(
        root_dir=None, name=None, k=None
):
    root = list(Path(root_dir, name,f"fold_{k}", "checkpoints").iterdir())[0]
    return str(root)


def load_model_path_by_args(args):
    return load_model_path(
        root_dir=args.log_dir,
        name=args.log_name,
        k=args.k,
    )
