"""Command-line interface for apertus_data."""

import argparse
import sys

from apertus_data.dataset import Dataset


def cmd_build(args: argparse.Namespace) -> None:
    dataset = Dataset.from_id(args.dataset_id)
    dataset.build(force=args.force)


def main() -> None:
    parser = argparse.ArgumentParser(prog='apertus_data', description='apertus_data CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    build_parser = subparsers.add_parser('build', help='Build a dataset by its catalogue id')
    build_parser.add_argument('dataset_id', help='Dataset id (e.g. owner___name___version)')
    build_parser.add_argument('--force', action='store_true', help='Wipe existing data and re-build')
    build_parser.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
