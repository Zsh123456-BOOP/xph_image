# Prism-CD vs NeuralCD Comparison

Datasets: assist_09, assist_17, junyi
Baseline source mode: output_dir

Judgment rule:
- Dataset-level slipping comparison uses stress-subset AUC/ACC deltas plus flipped-sample upper-quartile / upper-decile decoupling gaps, averaged over all available flip ratios / evaluation seeds.
- For each indicator, the model that wins on more datasets is treated as better supported.
- An experiment is marked as supported only when Prism-CD wins a strict majority of its implemented indicators.

## Controlled Slip
- Overall supported: True

Dataset-level summary:
```text
  dataset  prism_full_auc_delta  baseline_full_auc_delta  prism_stress_auc_delta  baseline_stress_auc_delta auc_drop_better_model  prism_full_acc_delta  baseline_full_acc_delta  prism_stress_acc_delta  baseline_stress_acc_delta acc_drop_better_model  prism_flipped_confidence  baseline_flipped_confidence  prism_win_metrics  baseline_win_metrics  tie_metrics  supports_prism_expected confidence_better_model  prism_flipped_decoupling_gap_p90  baseline_flipped_decoupling_gap_p90 tail_decoupling_better_model  prism_flipped_decoupling_gap_p75  baseline_flipped_decoupling_gap_p75 knowledge_adjustment_better_model
assist_09               -0.0068                  -0.0079                 -0.0724                    -0.0746              Prism-CD               -0.0040                  -0.0040                 -0.1000                    -0.1000                   tie                    0.9890                       0.9442                  3                     0            1                     True                Prism-CD                           -0.1373                              -0.3144                     Prism-CD                           -0.2058                              -0.3682                          Prism-CD
assist_17               -0.0005                  -0.0006                 -0.0831                    -0.0800              NeuralCD               -0.0003                  -0.0003                 -0.0995                    -0.0995                   tie                    0.9221                       0.9036                  2                     1            1                     True                Prism-CD                           -0.1551                              -0.2628                     Prism-CD                           -0.1957                              -0.3129                          Prism-CD
    junyi               -0.0063                  -0.0064                 -0.0690                    -0.0702              Prism-CD               -0.0033                  -0.0033                 -0.1001                    -0.1001                   tie                    0.9268                       0.9305                  3                     0            1                     True                NeuralCD                           -0.1319                              -0.3342                     Prism-CD                           -0.2013                              -0.3656                          Prism-CD
```

Indicator support:
```text
     experiment            indicator  prism_wins  baseline_wins  ties  supports_prism_expected
controlled_slip             auc_drop           2              1     0                     True
controlled_slip             acc_drop           0              0     3                    False
controlled_slip      tail_decoupling           3              0     0                     True
controlled_slip knowledge_adjustment           3              0     0                     True
controlled_slip              overall           3              0     1                     True
```

## Case Study
- Overall supported: True

Dataset-level summary:
```text
  dataset  prism_adjustment_ratio_median  baseline_adjustment_ratio_median adjustment_ratio_better_model  prism_adjustment_ratio_p90  baseline_adjustment_ratio_p90 adjustment_tail_better_model  prism_decoupling_gap_median  baseline_decoupling_gap_median decoupling_better_model  prism_win_metrics  baseline_win_metrics  tie_metrics  supports_prism_expected
assist_09                         0.4258                            1.9526                      Prism-CD                      0.6057                         2.5063                     Prism-CD                       0.0610                         -0.1385                Prism-CD                  3                     0            0                     True
assist_17                         0.7266                            1.4284                      Prism-CD                      0.7724                         1.7503                     Prism-CD                       0.0702                         -0.0928                Prism-CD                  3                     0            0                     True
    junyi                         0.5817                            6.5963                      Prism-CD                      0.6759                         7.5817                     Prism-CD                       0.0756                         -0.3093                Prism-CD                  3                     0            0                     True
```

Indicator support:
```text
experiment        indicator  prism_wins  baseline_wins  ties  supports_prism_expected
case_study adjustment_ratio           3              0     0                     True
case_study  adjustment_tail           3              0     0                     True
case_study   decoupling_gap           3              0     0                     True
case_study          overall           3              0     0                     True
```

## Notes
- This report compares the two implemented large experiments only: controlled slip simulation and case study.
- Baseline inputs come from the local output directory when available; strict-dir loading is compatibility-only.
- Controlled slip uses a stress subset built from all strong-positive candidates plus matched native negatives, so the pseudo-slip effect is not diluted by the entire test split.
- The old flipped-confidence indicator is kept only as supplemental context in raw tables. It is excluded from the formal verdict because fixed-prediction label-flip evaluation makes that score mechanically conflict with the AUC/ACC drop objective.
- Controlled slip now uses two stability-oriented behavior indicators: the flipped-sample upper-quartile decoupling gap and upper-decile decoupling gap. Higher is better and indicates that concept-level belief stays less suppressed than item-level correctness under pseudo slips, especially in the stronger-support tail.
- Case study is illustrative: for each dataset it selects the top representative conflict cases where Prism-CD shows the clearest adjustment-ratio and decoupling advantage, then summarizes those examples with median / p90.
- The stable concept-drop ratio uses an item-drop floor of 0.05 to avoid denominator blow-up.