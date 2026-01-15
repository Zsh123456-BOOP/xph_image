#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主控制脚本: 运行所有实验模块 (支持自动训练 + 智能 GPU 调度 + 并行执行 + 日志)
================================================================================
用法:
    # 运行所有数据集的所有实验（自动训练，自动选择 GPU，并行执行）
    python run_all_experiments.py

    # 只运行特定数据集
    python run_all_experiments.py --datasets assist_09,junyi

    # 只运行特定实验模块
    python run_all_experiments.py --modules 1,2

    # 串行执行（禁用并行）
    python run_all_experiments.py --serial

    # 手动指定 GPU（禁用自动选择）
    python run_all_experiments.py --no_auto_gpu --device cuda:0
================================================================================
"""

import os
import sys
import argparse
import subprocess
import time
import logging
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed


# 可用数据集
AVAILABLE_DATASETS = ["assist_09", "assist_17", "junyi"]

# 实验模块映射
EXPERIMENT_MODULES = {
    1: ("exp_module1_disentangle.py", "特征解耦实验"),
    2: ("exp_module2_gating_consistency.py", "鲁棒性与机制验证实验"),
    3: ("exp_module3_interaction_qnoise.py", "交互建模实验"),
}


# -----------------------------
# 日志配置
# -----------------------------
def setup_logging(log_dir: str) -> logging.Logger:
    """配置日志：同时输出到控制台和文件"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_all_experiments_{timestamp}.log")
    
    # 创建 logger
    logger = logging.getLogger("run_all_experiments")
    logger.setLevel(logging.DEBUG)
    
    # 清除已有 handlers
    logger.handlers = []
    
    # 文件 handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"日志文件: {log_file}")
    return logger


def format_duration(seconds: float) -> str:
    """格式化时长"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.2f}h"


# -----------------------------
# GPU 智能调度
# -----------------------------
def get_gpu_free_memory() -> List[Tuple[int, int]]:
    """获取所有 GPU 的可用显存（MB），按显存降序排列"""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            free_mb = free_mem // (1024 * 1024)
            gpu_info.append((i, free_mb))
        
        gpu_info.sort(key=lambda x: x[1], reverse=True)
        return gpu_info
    except Exception:
        return []


def select_best_gpu(min_free_mb: int = 4000) -> Optional[str]:
    """选择可用显存最多的 GPU"""
    gpu_info = get_gpu_free_memory()
    if not gpu_info:
        return None
    best_gpu, best_free = gpu_info[0]
    return f"cuda:{best_gpu}"


def get_available_gpus(min_free_mb: int = 3000) -> List[int]:
    """获取满足最低显存要求的 GPU 列表"""
    gpu_info = get_gpu_free_memory()
    if not gpu_info:
        return [0]
    available = [gpu_id for gpu_id, free_mb in gpu_info if free_mb >= min_free_mb]
    if not available:
        available = [gpu_info[0][0]]
    return available


def print_gpu_status(logger: logging.Logger):
    """打印 GPU 状态"""
    gpu_info = get_gpu_free_memory()
    if not gpu_info:
        logger.info("  [GPU] 无可用 CUDA 设备")
        return
    
    logger.info("  [GPU] 当前显存状态:")
    for gpu_id, free_mb in gpu_info:
        try:
            import torch
            _, total = torch.cuda.mem_get_info(gpu_id)
            total_mb = total // (1024 * 1024)
        except:
            total_mb = 0
        used_mb = total_mb - free_mb if total_mb > 0 else 0
        logger.info(f"    cuda:{gpu_id}: {free_mb:,}MB free / {total_mb:,}MB total")


# -----------------------------
# 参数解析
# -----------------------------
def get_args():
    p = argparse.ArgumentParser(description="主控制脚本: 运行所有实验模块")
    
    p.add_argument("--datasets", type=str, default=",".join(AVAILABLE_DATASETS),
                   help="要运行的数据集，逗号分隔")
    p.add_argument("--modules", type=str, default="1,2,3",
                   help="要运行的实验模块，逗号分隔")
    p.add_argument("--device", type=str, default=None,
                   help="手动指定 GPU 设备")
    p.add_argument("--no_auto_gpu", action="store_true",
                   help="禁用自动 GPU 选择")
    p.add_argument("--min_gpu_mem", type=int, default=4000,
                   help="最低可用显存要求（MB）")
    p.add_argument("--skip_train", action="store_true",
                   help="跳过自动训练")
    p.add_argument("--epochs", type=int, default=100,
                   help="训练 epochs")
    p.add_argument("--serial", action="store_true",
                   help="串行执行（禁用并行）")
    p.add_argument("--max_parallel", type=int, default=0,
                   help="最大并行数（0=自动，按 GPU 数量）")
    p.add_argument("--log_dir", type=str, default="logs",
                   help="日志目录")
    p.add_argument("--dry_run", action="store_true",
                   help="只打印命令，不执行")
    
    return p.parse_args()


def parse_datasets(s: str) -> List[str]:
    datasets = [d.strip() for d in s.split(",") if d.strip()]
    for d in datasets:
        if d not in AVAILABLE_DATASETS:
            raise ValueError(f"未知数据集: {d}")
    return datasets


def parse_modules(s: str) -> List[int]:
    modules = [int(m.strip()) for m in s.split(",") if m.strip()]
    for m in modules:
        if m not in EXPERIMENT_MODULES:
            raise ValueError(f"未知模块: {m}")
    return modules


def get_model_path(root: str, dataset: str) -> Tuple[str, bool]:
    """返回模型路径和是否存在"""
    model_path = os.path.join(root, "saved_models", dataset, "best_model.pth")
    if os.path.exists(model_path):
        return model_path, True
    model_path_alt = os.path.join(root, "saved_models", "best_model.pth")
    if os.path.exists(model_path_alt) and dataset == "assist_09":
        return model_path_alt, True
    return model_path, False


# -----------------------------
# 任务执行函数（可在子进程中运行）
# -----------------------------
def run_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行单个任务（训练或实验）
    task: {type, dataset, device, gpus, root, script, epochs, ...}
    返回: {task, success, duration, error}
    """
    start_time = time.time()
    result = {"task": task, "success": False, "duration": 0, "error": None}
    
    root = task["root"]
    dataset = task["dataset"]
    device = task["device"]
    
    try:
        if task["type"] == "train":
            # 训练模型
            data_dir = os.path.join(root, "data", dataset)
            save_dir = os.path.join(root, "saved_models", dataset)
            graph_dir = os.path.join(root, "graphs", dataset)
            
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(graph_dir, exist_ok=True)
            
            cmd = [
                sys.executable,
                os.path.join(root, "main.py"),
                "--train_file", os.path.join(data_dir, "train.csv"),
                "--valid_file", os.path.join(data_dir, "valid.csv"),
                "--test_file", os.path.join(data_dir, "test.csv"),
                "--graph_dir", graph_dir,
                "--save_dir", save_dir,
                "--device", device,
                "--epochs", str(task["epochs"]),
            ]
        else:
            # 运行实验
            script = task["script"]
            gpus = task.get("gpus", "0")
            
            cmd = [
                sys.executable,
                os.path.join(root, script),
                "--dataset", dataset,
                "--device", device,
            ]
            if "module2" in script:
                cmd.extend(["--gpus", gpus])
        
        # 执行命令
        proc = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        result["success"] = (proc.returncode == 0)
        if proc.returncode != 0:
            result["error"] = proc.stderr[-500:] if proc.stderr else "Unknown error"
            
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout (2h)"
    except Exception as e:
        result["error"] = str(e)
    
    result["duration"] = time.time() - start_time
    return result


# -----------------------------
# 主执行逻辑
# -----------------------------
def main():
    args = get_args()
    root = os.path.dirname(os.path.abspath(__file__))
    
    # 设置日志
    logger = setup_logging(args.log_dir)
    
    datasets = parse_datasets(args.datasets)
    modules = parse_modules(args.modules)
    
    # 获取可用 GPU
    available_gpus = get_available_gpus(args.min_gpu_mem)
    max_parallel = args.max_parallel if args.max_parallel > 0 else len(available_gpus)
    if args.serial:
        max_parallel = 1
    
    logger.info("="*60)
    logger.info("实验配置")
    logger.info("="*60)
    logger.info(f"数据集: {datasets}")
    logger.info(f"模块: {[f'Module-{m}' for m in modules]}")
    logger.info(f"自动训练: {'关闭' if args.skip_train else '开启'}")
    logger.info(f"并行模式: {'串行' if args.serial else f'并行 (max={max_parallel})'}")
    logger.info(f"可用 GPU: {available_gpus}")
    print_gpu_status(logger)
    logger.info("="*60)
    
    total_start = time.time()
    
    # ========================================
    # Phase 1: 检查并训练缺失的模型（并行）
    # ========================================
    if not args.skip_train:
        logger.info("\n" + "="*60)
        logger.info("Phase 1: 检查并训练缺失的模型")
        logger.info("="*60)
        
        train_tasks = []
        for i, dataset in enumerate(datasets):
            model_path, exists = get_model_path(root, dataset)
            if exists:
                logger.info(f"  [✓] {dataset}: 模型已存在")
            else:
                # 分配 GPU（轮询）
                gpu_id = available_gpus[i % len(available_gpus)]
                device = f"cuda:{gpu_id}" if not args.no_auto_gpu else (args.device or "cuda:0")
                
                train_tasks.append({
                    "type": "train",
                    "dataset": dataset,
                    "device": device,
                    "epochs": args.epochs,
                    "root": root,
                })
                logger.info(f"  [✗] {dataset}: 需要训练 (will use {device})")
        
        if train_tasks:
            phase1_start = time.time()
            logger.info(f"\n开始训练 {len(train_tasks)} 个模型...")
            
            if args.dry_run:
                logger.info("[DRY RUN] 跳过训练")
            else:
                # 并行训练
                with ProcessPoolExecutor(max_workers=min(max_parallel, len(train_tasks))) as executor:
                    futures = {executor.submit(run_single_task, t): t for t in train_tasks}
                    for future in as_completed(futures):
                        task = futures[future]
                        result = future.result()
                        status = "✓" if result["success"] else "✗"
                        duration = format_duration(result["duration"])
                        logger.info(f"  [{status}] {task['dataset']}: {duration}")
                        if result["error"]:
                            logger.error(f"    Error: {result['error'][:200]}")
            
            phase1_duration = time.time() - phase1_start
            logger.info(f"\nPhase 1 完成: {format_duration(phase1_duration)}")
    
    # ========================================
    # Phase 2: 运行实验（按模块分批，每批并行不同数据集）
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("Phase 2: 运行实验")
    logger.info("="*60)
    
    all_results = []
    
    for module_id in modules:
        script, desc = EXPERIMENT_MODULES[module_id]
        
        logger.info(f"\n--- Module-{module_id}: {desc} ---")
        module_start = time.time()
        
        # 构建该模块的所有数据集任务
        exp_tasks = []
        for i, dataset in enumerate(datasets):
            model_path, exists = get_model_path(root, dataset)
            if not exists:
                logger.warning(f"  [{dataset}] 模型不存在，跳过")
                all_results.append({
                    "module": module_id,
                    "dataset": dataset,
                    "success": False,
                    "duration": 0,
                    "error": "Model not found"
                })
                continue
            
            # 分配 GPU
            gpu_id = available_gpus[i % len(available_gpus)]
            device = f"cuda:{gpu_id}" if not args.no_auto_gpu else (args.device or "cuda:0")
            gpus = ",".join(str(g) for g in available_gpus)
            
            exp_tasks.append({
                "type": "experiment",
                "module": module_id,
                "dataset": dataset,
                "device": device,
                "gpus": gpus,
                "script": script,
                "root": root,
            })
            logger.debug(f"  Task: {dataset} -> {device}")
        
        if not exp_tasks:
            continue
        
        if args.dry_run:
            for t in exp_tasks:
                logger.info(f"  [DRY RUN] {t['dataset']} -> {t['device']}")
            continue
        
        # 并行执行该模块的所有数据集
        with ProcessPoolExecutor(max_workers=min(max_parallel, len(exp_tasks))) as executor:
            futures = {executor.submit(run_single_task, t): t for t in exp_tasks}
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                status = "✓" if result["success"] else "✗"
                duration = format_duration(result["duration"])
                logger.info(f"  [{status}] {task['dataset']}: {duration}")
                if result["error"]:
                    logger.error(f"    Error: {result['error'][:200]}")
                
                all_results.append({
                    "module": module_id,
                    "dataset": task["dataset"],
                    "success": result["success"],
                    "duration": result["duration"],
                    "error": result.get("error")
                })
        
        module_duration = time.time() - module_start
        logger.info(f"  Module-{module_id} 完成: {format_duration(module_duration)}")
    
    # ========================================
    # 汇总
    # ========================================
    total_duration = time.time() - total_start
    
    logger.info("\n" + "="*60)
    logger.info("实验汇总")
    logger.info("="*60)
    
    for m in modules:
        logger.info(f"\nModule-{m} ({EXPERIMENT_MODULES[m][1]}):")
        for r in all_results:
            if r["module"] == m:
                status = "✓ 成功" if r["success"] else "✗ 失败"
                dur = format_duration(r["duration"])
                logger.info(f"  [{r['dataset']}] {status} ({dur})")
    
    success_count = sum(1 for r in all_results if r["success"])
    total_count = len(all_results)
    
    logger.info(f"\n总计: {success_count}/{total_count} 成功")
    logger.info(f"总耗时: {format_duration(total_duration)}")
    logger.info("="*60)
    
    # 保存结果摘要
    import json
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_duration_sec": total_duration,
        "success_count": success_count,
        "total_count": total_count,
        "results": all_results,
    }
    summary_path = os.path.join(args.log_dir, "last_run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"结果摘要已保存: {summary_path}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except:
        pass
    main()
