"""Download a Roboflow Universe dataset in Ultralytics YOLO format.

Examples:
    python download_dataset.py --url https://universe.roboflow.com/<workspace>/<project>/dataset/<version> --api-key <KEY>
    python download_dataset.py --workspace <ws> --project <proj> --version 1 --api-key <KEY>

Requires the `roboflow` package (see training/requirements.txt) and an API
key from https://app.roboflow.com/settings/api. Pass it with --api-key or
set the ROBOFLOW_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

URL_PATTERN = re.compile(
    r"universe\.roboflow\.com/(?P<workspace>[^/]+)/(?P<project>[^/]+)"
    r"(?:/dataset/(?P<version>\d+))?"
)


def parse_url(url: str) -> tuple[str, str, int | None]:
    match = URL_PATTERN.search(url)
    if not match:
        raise ValueError(f"Could not parse a Roboflow Universe URL from: {url}")
    workspace, project, version = match.group("workspace", "project", "version")
    return workspace, project, int(version) if version else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="Roboflow Universe project URL")
    parser.add_argument("--workspace", help="Roboflow workspace slug (alternative to --url)")
    parser.add_argument("--project", help="Roboflow project slug (alternative to --url)")
    parser.add_argument("--version", type=int, default=1, help="Dataset version number (default: 1)")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"), help="Roboflow API key (or set ROBOFLOW_API_KEY)")
    parser.add_argument("--format", default="yolov11", help="Export format Roboflow should generate (default: yolov11)")
    parser.add_argument("--out", default=str(DATA_DIR), help="Destination directory (default: backend/training/data)")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("A Roboflow API key is required: pass --api-key or set ROBOFLOW_API_KEY.")

    if args.url:
        workspace, project, url_version = parse_url(args.url)
        version = url_version or args.version
    elif args.workspace and args.project:
        workspace, project, version = args.workspace, args.project, args.version
    else:
        sys.exit("Provide either --url or both --workspace and --project.")

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Missing dependency: pip install -r training/requirements.txt")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=args.api_key)
    rf_project = rf.workspace(workspace).project(project)
    rf_version = rf_project.version(version)
    dataset = rf_version.download(args.format, location=str(out_dir))

    print(f"Downloaded '{workspace}/{project}' v{version} ({args.format}) to {dataset.location}")
    print(f"data.yaml: {Path(dataset.location) / 'data.yaml'}")


if __name__ == "__main__":
    main()
