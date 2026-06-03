"""Aggregate COCO amplification results into summary tables."""

from __future__ import annotations

import argparse

from coco_amplification.run_experiment import aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    aggregate()


if __name__ == "__main__":
    main()
