#!/usr/bin/env python3
import argparse
from src.database.init_db import init_db
from src.utils.storage import LocalFileStorage
import shutil
from pathlib import Path
from src.config import settings

def main():
    parser = argparse.ArgumentParser(description="Management CLI")
    subparsers = parser.add_subparsers(dest="command")

    # DB init
    subparsers.add_parser("init-db", help="Initialize the database and insert sample data")

    # File cleanup
    cleanup_parser = subparsers.add_parser("cleanup-files", help="Delete old files from uploads dir")
    cleanup_parser.add_argument("--days", type=int, default=30, help="Retention days (default: 30)")

    # Model management
    model_parser = subparsers.add_parser("remove-models", help="Delete all downloaded models")

    args = parser.parse_args()
    if args.command == "init-db":
        init_db()
    elif args.command == "cleanup-files":
        storage = LocalFileStorage()
        deleted = storage.cleanup_old_files(retention_days=args.days)
        print(f"Deleted files: {deleted}")
    elif args.command == "remove-models":
        models_dir = Path(settings.models_dir)
        if models_dir.exists():
            shutil.rmtree(models_dir)
            print(f"Removed all models in {models_dir}")
        else:
            print(f"Models directory {models_dir} does not exist.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 