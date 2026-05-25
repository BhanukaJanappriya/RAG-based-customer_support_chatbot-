#!/usr/bin/env python3
"""CLI script for ingesting documents into the ChromaDB vector store.

Usage
-----
    python scripts/ingest.py                         # uses data/raw/
    python scripts/ingest.py --data-dir /path/to/docs
    python scripts/ingest.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDF and Markdown documents into the ChromaDB vector store."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Directory containing documents. Defaults to the value of RAW_DATA_DIR in .env.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import here so logging is configured first
    from app.ingestion.pipeline import run_ingestion

    try:
        stats = run_ingestion(args.data_dir)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logging.error(f"Ingestion failed: {exc}", exc_info=args.verbose)
        sys.exit(1)

    print("\n✓ Ingestion complete")
    print(f"  Documents loaded : {stats['documents_loaded']}")
    print(f"  Chunks created   : {stats['chunks_created']}")
    print(f"  Chunks stored    : {stats['chunks_stored']}")
    print(f"  Total in store   : {stats['total_in_store']}")


if __name__ == "__main__":
    main()
