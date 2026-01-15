#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Import experiment modules
import exp_module1_disentangle as m1
import exp_module2_gating_consistency as m2
import exp_module3_interaction_qnoise as m3
from utils import set_pub_style

def main():
    set_pub_style()
    root = "/home/zsh/xph_image"
    
    print(">>> Regenerating Experiment 1 Plots...")
    out_dir_m1 = os.path.join(root, "exp_m1_out")
    datasets = ["assist_09"] # Just test one for speed/verify
    # Or check if directories exist
    
    for ds in ["assist_09", "assist_17", "junyi"]:
        d_path = os.path.join(out_dir_m1, ds)
        if not os.path.exists(d_path):
            continue
        print(f"  [Exp1] Processing {ds}...")
        
        # 1. Combo
        try:
            m1.make_combo_from_csvs(d_path, leakage_thr=0.15)
            print(f"    -> Generated Combo plots in {d_path}/combo")
        except Exception as e:
            print(f"    [Err] Combo failed: {e}")

        # 2. Single features (replot if needed, but combo covers most)
        # We can try to load CSVs and call plot_alignment_leakage_and_specialists if we want single plots refreshed too.
        # R = pd.read_csv(...)
        # But make_combo_from_csvs uses separate plotting functions (plot_combo_*) that share style.
        # To update the single plots (alignment_leakage.png etc), we need to call plot_alignment_leakage_and_specialists.
        # But that function expects R matrix.
        # Let's verify combo mainly as it's the "new" target.
        # But user asked for "reading... exp_*_out ... if ugly modify".
        # So I should regenerate single plots too if possible.
        pass

    print("\n>>> Regenerating Experiment 2 Plots...")
    out_dir_m2 = os.path.join(root, "exp_m2_out")
    for ds in ["assist_09", "assist_17", "junyi"]:
        d_path = os.path.join(out_dir_m2, ds)
        if not os.path.exists(d_path):
            continue
        print(f"  [Exp2] Processing {ds}...")
        
        # Robust Curve
        path_robust = os.path.join(d_path, "robust_curve.csv")
        if os.path.exists(path_robust):
            df = pd.read_csv(path_robust)
            m2.plot_robust_curve_combo(df, os.path.join(d_path, "robust_curve.png"))
            print("    -> Regenerated robust_curve.png")
            
        # Pareto
        path_pareto = os.path.join(d_path, "pareto.csv")
        if os.path.exists(path_pareto):
            df = pd.read_csv(path_pareto)
            m2.plot_pareto_combo(df, os.path.join(d_path, "pareto.png"))
            print("    -> Regenerated pareto.png")

    print("\n>>> Regenerating Experiment 3 Plots...")
    out_dir_m3 = os.path.join(root, "exp_m3_out")
    for ds in ["assist_09", "assist_17", "junyi"]:
        d_path = os.path.join(out_dir_m3, ds)
        if not os.path.exists(d_path):
            continue
        print(f"  [Exp3] Processing {ds}...")
        
        # Q-noise Curve
        path_q = os.path.join(d_path, "qnoise_curve.csv")
        df_q = None
        if os.path.exists(path_q):
            df_q = pd.read_csv(path_q)
            m3.plot_qnoise_curve_single(df_q, os.path.join(d_path, "qnoise_curve.png"))
            print("    -> Regenerated qnoise_curve.png")
            
        # Hard Curve
        path_h = os.path.join(d_path, "qnoise_hard_curve.csv")
        df_h = None
        if os.path.exists(path_h):
            df_h = pd.read_csv(path_h)
            m3.plot_qnoise_hard_curve_single(df_h, os.path.join(d_path, "qnoise_hard_curve.png"))
            print("    -> Regenerated qnoise_hard_curve.png")
            
        # Combo
        if df_q is not None and df_h is not None:
            args_mock = SimpleNamespace(out_dir=d_path)
            # m3.plot_qnoise_combo(args_mock, df_q, df_h) # I need to check if I updated this function or if it was already there
            # I updated it to use COLORS? No, I skipped it in multi_replace because it wasn't in the snippet?
            # Wait, I need to check if plot_qnoise_combo was updated.
            # I viewed 'plot_qnoise_combo' in 'exp_module3' before?
            # Outline said 'plot_qnoise_combo' (447).
            # My multi_replace for exp3 edited 'plot_qnoise_curve_single' (new) and 'plot_qnoise_hard_curve_single' (new).
            # I did NOT edit 'plot_qnoise_combo' in the multi_replace call.
            # I must edit plot_qnoise_combo heavily to match style?
            # Or did I?
            # Let's check the file content first.
            pass

if __name__ == "__main__":
    main()
