"""Read config.json only. Does not import torch, weights, or the nano runner."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from .contracts import load_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.find_spec('nanovllm')
    expected = root / 'nanovllm' / '__init__.py'
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        raise RuntimeError('Wrong nanovllm import origin; run from the adaptation directory')
    result = load_contract(args.config)
    result['config_path'] = str(args.config.resolve())
    result['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    result['nanovllm_import_origin'] = str(expected)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
