#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主控制脚本: 运行所有实验模块
================================================================================
用法:
    # 运行所有数据集的所有实验
    python run_all_experiments.py

    # 只运行特定数据集
    python run_all_experiments.py --datasets assist_09,junyi

    # 只运行特定实验模块
    python run_all_experiments.py --modules 1,2

    # 组合使用
    python run_all_experiments.py --datasets assist_17 --modules 3 --device cuda:1
================================================================================
"""

import os
import sys
import argparse
import subprocess
from typing import List


# 可用数据集
AVAILABLE_DATASETS = ["assist_09", "assist_17", "junyi"]

# 实验模块映射
EXPERIMENT_MODULES = {
    1: ("exp_module1_disentangle.py", "特征解耦实验"),
    2: ("exp_module2_gating_consistency.py", "鲁棒性与机制验证实验"),
    3: ("exp_module3_interaction_qnoise.py", "交互建模实验"),
}


def get_args():
    p = argparse.ArgumentParser(description="主控制脚本: 运行所有实验模块")
    
    p.add_argument("--datasets", type=str, default=",".join(AVAILABLE_DATASETS),
                   help=f"要运行的数据集，逗号分隔。可选: {AVAILABLE_DATASETS}。默认全部运行。")
    
    p.add_argument("--modules", type=str, default="1,2,3",
                   help="要运行的实验模块，逗号分隔。可选: 1,2,3。默认全部运行。")
    
    p.add_argument("--device", type=str, default="cuda:0",
                   help="GPU 设备。默认: cuda:0")
    
    p.add_argument("--gpus", type=str, default="0,1,2,3",
                   help="Pareto 实验使用的 GPU 列表（仅 Module-2）。默认: 0,1,2,3")
    
    p.add_argument("--dry_run", action="store_true",
                   help="只打印要执行的命令，不实际运行")
    
    return p.parse_args()


def parse_datasets(s: str) -> List[str]:
    datasets = [d.strip() for d in s.split(",") if d.strip()]
    for d in datasets:
        if d not in AVAILABLE_DATASETS:
            raise ValueError(f"未知数据集: {d}。可选: {AVAILABLE_DATASETS}")
    return datasets


def parse_modules(s: str) -> List[int]:
    modules = [int(m.strip()) for m in s.split(",") if m.strip()]
    for m in modules:
        if m not in EXPERIMENT_MODULES:
            raise ValueError(f"未知模块: {m}。可选: {list(EXPERIMENT_MODULES.keys())}")
    return modules


def run_experiment(script: str, dataset: str, device: str, gpus: str, dry_run: bool, root: str) -> bool:
    """运行单个实验脚本"""
    data_dir = os.path.join(root, "data", dataset)
    
    # 检查数据集是否存在
    if not os.path.exists(data_dir):
        print(f"[WARN] 数据集目录不存在: {data_dir}，跳过。")
        return False
    
    # 检查模型是否存在
    model_path = os.path.join(root, "saved_models", dataset, "best_model.pth")
    if not os.path.exists(model_path):
        # 兼容旧结构：尝试默认路径
        model_path_alt = os.path.join(root, "saved_models", "best_model.pth")
        if os.path.exists(model_path_alt) and dataset == "assist_09":
            model_path = model_path_alt
        else:
            print(f"[WARN] 模型不存在: {model_path}，跳过。请先训练模型。")
            return False
    
    # 构建命令
    cmd = [
        sys.executable,
        os.path.join(root, script),
        "--dataset", dataset,
        "--device", device,
    ]
    
    # Module-2 特殊参数
    if "module2" in script:
        cmd.extend(["--gpus", gpus])
    
    print(f"\n{'='*60}")
    print(f"[RUN] {script} | Dataset: {dataset}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"{'='*60}")
    
    if dry_run:
        print("[DRY RUN] 跳过实际执行")
        return True
    
    try:
        result = subprocess.run(cmd, check=True, cwd=root)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script} 执行失败: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 意外错误: {e}")
        return False


def main():
    args = get_args()
    root = os.path.dirname(os.path.abspath(__file__))
    
    datasets = parse_datasets(args.datasets)
    modules = parse_modules(args.modules)
    
    print("="*60)
    print("实验配置")
    print("="*60)
    print(f"数据集: {datasets}")
    print(f"模块: {[f'Module-{m}: {EXPERIMENT_MODULES[m][1]}' for m in modules]}")
    print(f"设备: {args.device}")
    print(f"GPUs (Module-2): {args.gpus}")
    print(f"Dry Run: {args.dry_run}")
    print("="*60)
    
    results = []
    
    for dataset in datasets:
        for module_id in modules:
            script, desc = EXPERIMENT_MODULES[module_id]
            success = run_experiment(
                script=script,
                dataset=dataset,
                device=args.device,
                gpus=args.gpus,
                dry_run=args.dry_run,
                root=root
            )
            results.append({
                "dataset": dataset,
                "module": module_id,
                "desc": desc,
                "success": success
            })
    
    # 汇总结果
    print("\n" + "="*60)
    print("实验汇总")
    print("="*60)
    for r in results:
        status = "✓ 成功" if r["success"] else "✗ 失败/跳过"
        print(f"  [{r['dataset']}] Module-{r['module']} ({r['desc']}): {status}")
    
    success_count = sum(1 for r in results if r["success"])
    print(f"\n总计: {success_count}/{len(results)} 成功")
    print("="*60)


if __name__ == "__main__":
    main()
