import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from analysis.hparam_sensitivity_utils import build_prism_hparam_sweep_jobs


def parse_list(raw):
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run xph_image Prism-CD hyperparameter sensitivity sweeps.")
    parser.add_argument("--datasets", type=str, default="assist_09,assist_17,junyi")
    parser.add_argument("--hparams", type=str, default="ortho_weight,dropout,embedding_dim")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--allowed_gpus", type=str, default="2,3")
    parser.add_argument("--max_concurrent_jobs", type=int, default=1)
    parser.add_argument("--cooldown_seconds", type=int, default=20)
    parser.add_argument("--wait_seconds", type=int, default=20)
    parser.add_argument("--memory_threshold_mb", type=int, default=2000)
    parser.add_argument("--result_csv", type=str, default="results/xph_image_hparam_results.csv")
    parser.add_argument("--job_result_dir", type=str, default="results/hparam_job_results")
    parser.add_argument("--save_root", type=str, default="saved_models/hparam_runs")
    parser.add_argument("--graph_root", type=str, default="graphs_hparam")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--skip_existing", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dry_run", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def get_gpu_memory_usage():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return {}

    usage = {}
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        usage[int(parts[0])] = int(parts[1])
    return usage


def choose_gpu(running_procs, allowed_gpus, memory_threshold_mb, max_concurrent_jobs):
    gpu_load = {gpu_id: 0 for gpu_id in allowed_gpus}
    for _, _, gpu_id in running_procs:
        if gpu_id in gpu_load:
            gpu_load[gpu_id] += 1

    usage = get_gpu_memory_usage()
    for gpu_id in allowed_gpus:
        if gpu_load[gpu_id] >= max_concurrent_jobs:
            continue
        if usage and usage.get(gpu_id, 10**9) > memory_threshold_mb:
            continue
        return gpu_id
    return None


def build_command(job, args):
    cfg = job["config"]
    dataset = job["dataset"]
    tag = job["tag"]
    save_dir = Path(args.save_root) / dataset / tag
    graph_dir = Path(args.graph_root) / dataset
    output_json = Path(args.job_result_dir) / f"{tag}.json"
    dataset_dir = Path(args.data_root) / dataset
    cmd = [
        sys.executable,
        "analysis/run_train_eval_job.py",
        "--dataset",
        dataset,
        "--train_file",
        str(dataset_dir / "train.csv"),
        "--valid_file",
        str(dataset_dir / "valid.csv"),
        "--test_file",
        str(dataset_dir / "test.csv"),
        "--graph_dir",
        str(graph_dir),
        "--save_dir",
        str(save_dir),
        "--output_json",
        str(output_json),
        "--tag",
        tag,
        "--seed",
        str(job["seed"]),
    ]
    for key, value in cfg.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd, output_json


def collect_existing_tags(job_result_dir):
    result_dir = Path(job_result_dir)
    if not result_dir.exists():
        return set()
    return {path.stem for path in result_dir.glob("*.json")}


def combine_job_results(job_result_dir, result_csv):
    result_dir = Path(job_result_dir)
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {key: value for key, value in payload.items() if key != "Config"}
        row.update(payload.get("Config", {}))
        rows.append(row)
    frame = pd.DataFrame(rows)
    Path(result_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(result_csv, index=False)
    return frame


def main():
    args = parse_args()
    datasets = set(parse_list(args.datasets))
    hparams = set(parse_list(args.hparams))
    allowed_gpus = [int(v) for v in parse_list(args.allowed_gpus)]
    Path(args.job_result_dir).mkdir(parents=True, exist_ok=True)

    jobs = [
        job
        for job in build_prism_hparam_sweep_jobs(seed=args.seed)
        if job["dataset"] in datasets and job["hparam"] in hparams
    ]
    existing_tags = collect_existing_tags(args.job_result_dir) if int(args.skip_existing) else set()
    pending_jobs = [job for job in jobs if job["tag"] not in existing_tags]

    print(f"Total sweep jobs: {len(jobs)}")
    print(f"Pending jobs after skip_existing: {len(pending_jobs)}")

    if int(args.dry_run):
        for job in pending_jobs:
            cmd, output_json = build_command(job, args)
            print(job["tag"], output_json)
            print(" ".join(cmd))
        return

    running_procs = []
    while pending_jobs or running_procs:
        for proc_info in running_procs[:]:
            proc, tag, gpu_id = proc_info
            if proc.poll() is not None:
                print(f"Finished {tag} on GPU {gpu_id} with code {proc.returncode}")
                running_procs.remove(proc_info)

        while pending_jobs:
            gpu_id = choose_gpu(
                running_procs=running_procs,
                allowed_gpus=allowed_gpus,
                memory_threshold_mb=args.memory_threshold_mb,
                max_concurrent_jobs=max(int(args.max_concurrent_jobs), 1),
            )
            if gpu_id is None:
                break

            job = pending_jobs.pop(0)
            cmd, output_json = build_command(job, args)
            if output_json.exists():
                print(f"Skipping existing {job['tag']}")
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Launching {job['tag']} on GPU {gpu_id}")
            proc = subprocess.Popen(cmd, env=env, cwd=os.getcwd())
            running_procs.append((proc, job["tag"], gpu_id))
            time.sleep(max(int(args.cooldown_seconds), 0))

        if pending_jobs or running_procs:
            time.sleep(max(int(args.wait_seconds), 1))

    result_frame = combine_job_results(args.job_result_dir, args.result_csv)
    print(f"Saved combined results to {args.result_csv} ({len(result_frame)} rows)")


if __name__ == "__main__":
    main()
