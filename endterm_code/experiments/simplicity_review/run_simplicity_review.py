#!/usr/bin/env python3
"""Run the frozen TRAIN-only protocol; refuse existing output directories."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.project_source.resolve()
    sys.path.insert(0, str(source))
    from simplicity_review import run_review
    result = run_review(source, args.output_dir)
    print(result['selection']['decision'] + ': ' + result['selection']['selected_name'])


if __name__ == '__main__':
    main()
