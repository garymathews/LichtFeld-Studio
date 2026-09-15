#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Assert two identical pinned-SOURCE_DATE_EPOCH training runs produce byte-identical checkpoints.

The identity seed includes the destination path (ledger section 77), so both runs must write to the *same*
output directory; the artifact is copied out after each run and the two copies are compared byte-for-byte.
The epoch seeds those derived identities, so a printed digest is comparable only with another run at the same
epoch - two runs at the same epoch agree, runs at different epochs legitimately differ (ledger section 117).
Exits non-zero on any difference.
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def default_epoch():
    """A SOURCE_DATE_EPOCH guaranteed ahead of any timestamp the runs will embed.

    The checkpoint serializer rejects created_at/modified_at that are not monotonic, so the
    epoch must exceed both the source files' mtimes and the resume checkpoint's embedded
    timestamps. Wall clock + one day satisfies that for any checkpoint made with a past or
    current epoch; pass --epoch explicitly to exceed a checkpoint created with a future pin.
    """
    return int(time.time()) + 86400


def sha256(path):
    with path.open("rb") as stream:
        if hasattr(hashlib, "file_digest"):  # Python 3.11+
            return hashlib.file_digest(stream, "sha256").hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build-macos-app/LichtFeld-Studio")
    parser.add_argument("--dataset", required=True, help="training dataset directory")
    parser.add_argument("--resume", required=True, help="checkpoint to resume from (.licht)")
    parser.add_argument("--iterations", type=int, default=8030)
    parser.add_argument("--epoch", type=int, default=None,
                        help="SOURCE_DATE_EPOCH pin; defaults to now + 1 day (pass explicitly to exceed a future-pinned checkpoint)")
    parser.add_argument("--max-cap", type=int, default=1000000)
    parser.add_argument("--resize", default="1")
    parser.add_argument("--work", default="/tmp/lfs-reproducibility")
    parser.add_argument("--icd", default="build-macos-app/resources/vulkan/MoltenVK_icd.json")
    args = parser.parse_args()

    work = Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    shared = work / "out"  # the *same* path for both runs - see the module docstring

    epoch = args.epoch if args.epoch is not None else default_epoch()
    print(f"SOURCE_DATE_EPOCH={epoch}")
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch))
    icd = Path(args.icd)
    if icd.exists():
        env["VK_ICD_FILENAMES"] = str(icd.resolve())

    artifacts = []
    for trial in ("a", "b"):
        shutil.rmtree(shared, ignore_errors=True)
        shared.mkdir(parents=True)
        command = [str(Path(args.binary).resolve()), "--headless", "--safe-mode", "--no-download",
                   "-d", str(Path(args.dataset).resolve()), "-o", str(shared),
                   "--iter", str(args.iterations), "--max-cap", str(args.max_cap), "-r", str(args.resize),
                   "--resume", str(Path(args.resume).resolve())]
        with (work / f"{trial}.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"FAIL: run {trial} exited {result.returncode}; see {work / f'{trial}.log'}")
            return 1
        artifact = shared / "project.licht"
        if not artifact.exists():
            print(f"FAIL: run {trial} wrote no project.licht; see {work / f'{trial}.log'}")
            return 1
        copy = work / f"{trial}.licht"
        shutil.copy2(artifact, copy)
        artifacts.append(copy)

    first, second = artifacts
    if first.stat().st_size != second.stat().st_size:
        print(f"FAIL: artifact sizes differ ({first.stat().st_size} vs {second.stat().st_size})")
        return 1
    first_digest, second_digest = sha256(first), sha256(second)
    if first_digest != second_digest:
        print(f"FAIL: artifacts differ\n  run a: {first_digest}\n  run b: {second_digest}")
        return 1
    print(f"OK: byte-identical checkpoints, {first.stat().st_size} bytes, epoch {epoch}, sha256 {first_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
