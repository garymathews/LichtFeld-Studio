#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run alternating, isolated Vulkan training trials; retain inputs and raw timings."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


# Two-tailed 95% t critical values by degrees of freedom (trials - 1);
# beyond the table the normal approximation is close enough for this purpose.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def summarise(report):
    steps = report.get("step_ms") or []
    ordered = sorted(steps)
    return {
        "steady_ms_per_iter": report["steady_ms_per_iter"],
        "warmup_steps": report["warmup_steps"],
        "steady_steps": report["steady_steps"],
        "median_ms": statistics.median(steps) if steps else None,
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)] if steps else None,
        "last_loss": report["last_loss"],
        "last_live_splats": report["last_live_splats"],
    }


def checkpoint_digest(run):
    """Content digest of the run's saved checkpoint, or None when it wrote none.

    Reported for every run, but not used to decide equivalence. Runs of one build are not
    bit-reproducible on this backend: the gradient passes accumulate through concurrent float
    atomics, so accumulation order varies with device scheduling and the encoded parameters
    differ in their low bits even with the seeds fixed and no random source in the measured
    path. A digest therefore *documents* the limitation and gives a future deterministic
    backend something to assert on; it must not be read as a failure when it differs.
    """
    path = run / "training/project.licht"
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size,
            "payload": payload_digest(path)}


def payload_digest(path, metadata_bytes=1 << 16):
    """Digest of the checkpoint *payload*, ignoring per-run metadata and the trailing checksum.

    A whole-file digest can never match: the header carries per-run metadata (this codebase builds
    UUIDs from `random_device`) and the trailer checksums the payload. Measured on a 687 MB
    checkpoint whose training trajectory is deterministic, whole-file differences collapse to
    ~0.002% of bytes while the payload is what matters - so this is what "identical checkpoints"
    has to mean.
    """
    size = path.stat().st_size
    if size <= 2 * metadata_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(metadata_bytes)
        remaining = size - 2 * metadata_bytes
        while remaining > 0:
            block = stream.read(min(1 << 20, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return {"sha256": digest.hexdigest(), "bytes": size - 2 * metadata_bytes,
            "skipped_head": metadata_bytes, "skipped_tail": metadata_bytes}


def byte_difference_summary(path_a, path_b, buckets=64):
    """Where two checkpoint files differ: count, extent, and a coarse histogram.

    Reported because the payload digest cannot pass by construction: the container records a
    per-run project UUID in its header and index, and a checksum table over those regions, so two
    runs of identical content always differ in a few thousand bytes. Having the *pattern* of the
    difference - a handful of bytes at the ends against a spread across the middle - is what tells
    a content difference apart from recorded identity.
    """
    size_a, size_b = path_a.stat().st_size, path_b.stat().st_size
    if size_a != size_b:
        return {"comparable": False, "size_a": size_a, "size_b": size_b}
    offsets = []
    with path_a.open("rb") as fa, path_b.open("rb") as fb:
        block = 1 << 20
        base = 0
        while True:
            ba, bb = fa.read(block), fb.read(block)
            if not ba:
                break
            if ba != bb:
                for i in range(len(ba)):
                    if ba[i] != bb[i]:
                        offsets.append(base + i)
            base += len(ba)
    hist = [0] * buckets
    for o in offsets:
        hist[min(buckets - 1, o * buckets // max(size_a, 1))] += 1
    return {"comparable": True, "size": size_a, "differing_bytes": len(offsets),
            "fraction": len(offsets) / max(size_a, 1),
            "first_offset": offsets[0] if offsets else None,
            "last_offset": offsets[-1] if offsets else None,
            "histogram": hist}


def equivalence(reference, candidate, loss_tolerance):
    """Screen `candidate` against the reference. This cannot prove a change exact.

    Splat counts must match exactly - they are discrete and independent of float noise, so a
    mismatch means the runs did different work. Loss is only a screen: measured here, two runs of
    *one* build can land in either of two outcome branches 6.6e-4 apart (a discrete decision
    flipping on accumulation order), while runs inside one branch agree to ~1e-6. No single
    tolerance therefore separates "same build" from "different build" on final loss.

    Proving a change exact needs an RNG-free in-process test on a fixed model, which is
    deterministic because it stays on the single-region path - the pattern used for the retained
    gradient-clear change. Checkpoint digests are recorded too, but never match: see
    checkpoint_digest(). Both fields here come from the run's own perf_bench.json.
    """
    splats_match = reference["last_live_splats"] == candidate["last_live_splats"]
    delta = candidate["last_loss"] - reference["last_loss"]
    ref_payload = (reference.get("checkpoint") or {}).get("payload")
    cand_payload = (candidate.get("checkpoint") or {}).get("payload")
    payload_match = bool(ref_payload and cand_payload and
                         ref_payload["sha256"] == cand_payload["sha256"])
    # The verdict rests on the training outcome, not on a file digest: content-identical runs
    # still differ in recorded identity, so a digest can only ever report "different". Losses are
    # compared bitwise first - the strongest signal available - with the tolerance as a fallback
    # for builds that differ deliberately in numerics.
    losses_bitwise = reference["last_loss"] == candidate["last_loss"]
    return {
        "live_splats_match": splats_match,
        "loss_delta": delta,
        "losses_bitwise_identical": losses_bitwise,
        "loss_within_tolerance": abs(delta) <= loss_tolerance,
        "payload_matches_reference": payload_match,
        "equivalent": bool(splats_match and (losses_bitwise or abs(delta) <= loss_tolerance)),
    }


def report_results(output, variants, loss_tolerance):
    """Per-run summaries plus mean paired deltas against the first variant."""
    names = [name for name, _ in variants]
    trials = {}
    for path in sorted(output.iterdir()):
        if not path.is_dir():
            continue
        for name in names:
            trial, separator, _ = path.name.partition(f"-{name}")
            report = path / "training/perf_bench.json"
            if not separator or not trial.isdigit() or not report.exists():
                continue
            summary = summarise(json.loads(report.read_text()))
            summary["checkpoint"] = checkpoint_digest(path)
            trials.setdefault(int(trial), {})[name] = summary
    completed = {trial: row for trial, row in sorted(trials.items()) if set(row) == set(names)}
    paired = {}
    # Calibrated comparison of the artifacts themselves. Measured on this format: identical
    # training with different recorded identity differs in ~0.0016% of bytes (header UUIDs, an
    # index UUID, and the checksum table over them), while genuinely different training changes the
    # file *size*, because the container compresses its payload. So size equality plus a byte
    # difference under a tenth of a percent is a strong "same training, different recorded
    # identity" signal - and a payload digest could never report "same" at all.
    checkpoint_comparison = {}
    for trial, row in completed.items():
        first = sorted(output.glob(f"{trial:02d}-*/training/project.licht"))
        if not first:
            continue
        for name in names[1:]:
            others = sorted(output.glob(f"{trial:02d}-{name}/training/project.licht"))
            if not others:
                continue
            # No percentage heuristic here: calibration on a 661 MB artifact gave 0.0016% and on a
            # 6 MB one 0.0059%, because the identity regions are fixed-size while the payload is not.
            # The calibrated signals are the ones the verdict uses - equal size, identical splat
            # counts, bitwise-identical losses - and this summary is diagnostic only.
            checkpoint_comparison[f"{trial:02d}:{name}"] = byte_difference_summary(first[0], others[0])
    matched = {}
    for trial, row in completed.items():
        reference = row[names[0]]
        matched[f"{trial:02d}"] = {name: equivalence(reference, row[name], loss_tolerance)
                                   for name in names[1:]}
    for name in names[1:]:
        deltas = [row[name]["steady_ms_per_iter"] - row[names[0]]["steady_ms_per_iter"] for row in completed.values()]
        mean = statistics.fmean(deltas)
        dof = len(deltas) - 1
        half = T95.get(dof, 1.96) * statistics.stdev(deltas) / math.sqrt(len(deltas)) if dof > 0 else float("nan")
        relative = mean / statistics.fmean(row[names[0]]["steady_ms_per_iter"] for row in completed.values())
        paired[f"{name}_minus_{names[0]}"] = {
            "trials": len(deltas), "per_trial_ms": deltas, "mean_ms": mean, "ci95_ms": [mean - half, mean + half],
            "relative": relative, "t95": T95.get(dof, 1.96),
        }
    (output / "results.json").write_text(json.dumps(
        {"runs": {f"{trial:02d}-{name}": row[name] for trial, row in completed.items() for name in names},
         "paired": paired,
         "equivalence": {"loss_tolerance": loss_tolerance, "per_trial": matched},
         "checkpoint_comparison": checkpoint_comparison,
         "note": ("equivalence compares live splat counts exactly and loss within the measured "
                  "run-to-run floor; checkpoint digests are recorded but are not expected to match, "
                  "because accumulation order over concurrent float atomics is not reproducible")},
        indent=2) + "\n")


def command_output(command):
    try:
        # Version probes (notably `ld -v`) report on stderr.
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return lines[0].strip() if lines else None


def toolchain():
    """Compiler/linker identity. A new SDK in front of an old linker is invisible in build flags."""
    ld = command_output(["/usr/bin/clang++", "-print-prog-name=ld"])
    return {
        "compiler": command_output(["/usr/bin/c++", "--version"]),
        "linker": command_output([ld, "-v"]) if ld else None,
        "sdk": command_output(["xcrun", "--show-sdk-path"]),
        "os": command_output(["sw_vers", "-buildVersion"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, help="NAME=/absolute/path/to/executable")
    parser.add_argument("--variant-env", action="append", default=[], metavar="NAME=KEY=VALUE",
                        help="Extra environment variable for one variant (repeatable)")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--warmup", type=int, default=200,
                        help="Exclude through this absolute iteration; for resume, add checkpoint iteration to warmup length")
    parser.add_argument("--resize", choices=["1", "2", "4", "8"], default="1")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--loss-tolerance", type=float, default=1e-3,
                        help=("absolute loss difference treated as equivalent; a screen, not a proof. "
                              "Measured on this backend: two runs of one build can land in either of two "
                              "outcome branches 6.6e-4 apart, so a tolerance tight enough to separate "
                              "builds from each other does not exist for final loss alone"))
    parser.add_argument("--lpips-weights", type=Path)
    args = parser.parse_args()
    variants = [item.split("=", 1) for item in args.variant]
    if any(len(v) != 2 or not v[0] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in v[0]) for v in variants):
        parser.error("Variants must be NAME=EXECUTABLE, with alphanumeric names (also '-' or '_')")
    if args.trials < 1 or args.warmup < 1 or len({v[0] for v in variants}) != len(variants):
        parser.error("Use positive trials/warmup and unique variant names")
    args.output = args.output.resolve()
    args.dataset = args.dataset.resolve(strict=True)
    variant_env = {}
    for item in args.variant_env:
        name, separator, assignment = item.partition("=")
        key, separator2, value = assignment.partition("=")
        if not separator or not separator2 or not name or not key:
            parser.error(f"--variant-env expects NAME=KEY=VALUE, got {item!r}")
        variant_env.setdefault(name, {})[key] = value
    if unknown := sorted(set(variant_env) - {v[0] for v in variants}):
        parser.error(f"--variant-env names unknown variants: {', '.join(unknown)}")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"arguments": vars(args), "variant_env": variant_env, "platform": platform.platform(),
                "toolchain": toolchain(), "variants": {}, "dataset_files": {}}
    for name, executable in variants:
        executable = Path(executable).resolve(strict=True)
        files = [executable, *executable.parent.glob("*.dylib"), *executable.parent.glob("resources/**/*.spv")]
        hashes = {}
        for path in files:
            with path.open("rb") as stream:
                hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["variants"][name] = hashes
    for path in sorted(args.dataset.rglob("*")):
        if path.is_file():
            manifest["dataset_files"][str(path.relative_to(args.dataset))] = [path.stat().st_size, path.stat().st_mtime_ns]
    if args.resume:
        with args.resume.open("rb") as stream:
            manifest["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    if args.lpips_weights:
        with args.lpips_weights.open("rb") as stream:
            manifest["lpips_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n")
    for trial in range(args.trials):
        for name, executable in variants[::1 if trial % 2 == 0 else -1]:
            run = args.output / f"{trial + 1:02d}-{name}"
            run.mkdir()
            command = [str(Path(executable).resolve()), "--headless", "--safe-mode", "--no-download",
                       "-d", str(args.dataset.resolve()), "-o", str(run / "training"),
                       "--iter", str(args.iterations), "--max-cap", "1000000", "-r", args.resize,
                       "--perf-bench", "--perf-bench-warmup", str(args.warmup)]
            if args.resume:
                command += ["--resume", str(args.resume.resolve())]
            if args.evaluate:
                command += ["--eval", "--eval-steps", str(args.iterations)]
            # variant-env may override LFS_HOME; passing it as a keyword as well would raise
            # "got multiple values for keyword argument".
            env = dict(os.environ, LFS_HOME=str(run / "home"))
            env.update(variant_env.get(name, {}))
            if args.lpips_weights:
                env["LFS_LPIPS_WEIGHTS"] = str(args.lpips_weights.resolve())
            start = time.monotonic()
            with (run / "console.log").open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            record = {"command": command, "env": variant_env.get(name, {}),
                      "elapsed_s": time.monotonic() - start, "returncode": result.returncode}
            (run / "run.json").write_text(json.dumps(record, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(f"{run} failed; see console.log")
            report = json.loads((run / "training/perf_bench.json").read_text())
            if not report["steady_steps"]:
                raise RuntimeError(f"{run} has no measured iterations")
            digest = checkpoint_digest(run)
            print(f"{run.name}: {report['steady_ms_per_iter']:.3f} ms/iteration, loss={report['last_loss']:.7f}"
                  f", splats={report['last_live_splats']}"
                  f", ckpt={(digest or {}).get('sha256', 'none')[:12]}", flush=True)
    report_results(args.output, variants, args.loss_tolerance)


if __name__ == "__main__":
    main()
