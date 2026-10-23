"""Download the author's OmniSF and component weights into shared storage.

Install modelscope>=1.38 in the omnisf Conda environment. Authentication is read
from MODELSCOPE_API_TOKEN or --prompt-token; it is never saved in the manifest.
Rerunning uses the ModelScope SDK's existing-file and resume handling.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import fnmatch
import getpass
import hashlib
import json
import os
from pathlib import Path
import time

from modelscope_hub import HubApi

REPOSITORIES = {
    "omnisf": ("haofuly/omnisf", "dataset", ["omnisf_9.0_jax_35999/*"]),
    "components": ("haofuly/omnisf-Components-ckpts", "model", None),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def download_one(name, root, endpoint, token, workers, attempts, retry_delay):
    repo, kind, patterns = REPOSITORIES[name]
    destination = root / repo.split("/", 1)[1]
    status_file = root / ".downloads" / f"{name}.json"
    status = {
        "repo": repo,
        "repo_type": kind,
        "include": patterns,
        "destination": str(destination),
        "endpoint": endpoint,
        "status": "pending",
    }
    for attempt in range(1, attempts + 1):
        status.update(status="downloading", attempt=attempt)
        write_json(status_file, status)
        try:
            api = HubApi(endpoint=endpoint, token=token)
            files = [
                file
                for file in api.list_repo_files(repo, kind, revision="master")
                if not file.is_dir and (patterns is None or any(fnmatch.fnmatchcase(file.path, p) for p in patterns))
            ]
            if not files:
                raise ValueError(f"No matching files in {repo}")
            status["files"] = [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in files]
            status["bytes"] = sum(f.size for f in files)
            write_json(status_file, status)
            print(f"{name}: {len(files)} files, {status['bytes'] / 2**30:.2f} GiB -> {destination}", flush=True)
            api.download_repo(
                repo, kind, revision="master", local_dir=destination, allow_patterns=patterns, max_workers=workers
            )
            status["status"] = "verifying"
            write_json(status_file, status)
            checked_hashes = 0
            for file in files:
                path = destination / file.path
                if not path.is_file() or path.stat().st_size != file.size:
                    raise ValueError(f"Missing or wrong-sized download: {path}")
                if file.sha256:
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if digest != file.sha256:
                        raise ValueError(f"SHA256 mismatch: {path}")
                    checked_hashes += 1
            status.update(status="complete", verified_files=len(files), verified_sha256=checked_hashes)
            status.pop("error", None)
            write_json(status_file, status)
            print(f"{name}: download and verification complete", flush=True)
            return True
        except Exception as error:
            message = str(error).replace(token, "[REDACTED]") if token else str(error)
            status.update(status="retrying" if attempt < attempts else "failed", error=message)
            write_json(status_file, status)
            print(f"{name}: attempt {attempt}/{attempts}: {type(error).__name__}: {message}", flush=True)
            if attempt < attempts:
                time.sleep(retry_delay)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("/share/project/robot-rfm/robot-model"))
    parser.add_argument("--endpoint", default="https://www.modelscope.cn")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=20)
    parser.add_argument("--only", choices=["both", *REPOSITORIES], default="both")
    parser.add_argument("--prompt-token", action="store_true")
    parser.add_argument("--direct", action="store_true", help="Disable proxies for this process only.")
    args = parser.parse_args()
    if args.max_workers < 1 or args.attempts < 1 or args.retry_delay < 0:
        parser.error("Workers and attempts must be positive; retry delay must be nonnegative.")
    if args.direct:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(name, None)
    token = (
        getpass.getpass("ModelScope token (hidden): ")
        if args.prompt_token
        else os.environ.get("MODELSCOPE_API_TOKEN", "")
    )
    selected = list(REPOSITORIES) if args.only == "both" else [args.only]
    results = []
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = [
            executor.submit(
                download_one,
                name,
                args.model_root.resolve(),
                args.endpoint,
                token,
                args.max_workers,
                args.attempts,
                args.retry_delay,
            )
            for name in selected
        ]
        results.extend(future.result() for future in as_completed(futures))
    if not all(results):
        raise SystemExit(
            "Download incomplete. See <model-root>/.downloads/*.json and rerun after connectivity is restored."
        )


if __name__ == "__main__":
    main()
