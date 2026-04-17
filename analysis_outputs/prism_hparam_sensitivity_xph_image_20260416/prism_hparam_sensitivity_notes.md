# Prism-CD 参数敏感性实验说明

本文档汇总 Prism-CD 在三个数据集上的单因子参数敏感性实验结果。

## Hyperparameters

- `ortho_weight`
- `dropout`
- `embedding_dim`

## Main Findings

- 按平均 AUC 增益看，`dropout` 的波动最明显，其平均 AUC 增益为 0.0015。
- 单项最大 AUC 增益出现在 `assist_09` 的 `dropout`，默认值到最优值带来 0.0028 的 AUC 提升。
- 单项最大 ACC 增益出现在 `junyi` 的 `dropout`，默认值到最优值带来 0.0037 的 ACC 提升。
- 相对默认配置，发生非默认最佳点的数据集/参数组合共有 7 个：assist_09/dropout -> 0.2, assist_09/embedding_dim -> 384, assist_09/ortho_weight -> 0.3, assist_17/ortho_weight -> 1, junyi/dropout -> 0.2, junyi/embedding_dim -> 384, junyi/ortho_weight -> 0.7。
- 当前 sweep 中默认值已经等于最佳点的组合有 2 个：assist_17/dropout, assist_17/embedding_dim。
- 整体看，三个超参数带来的收益都不大，说明正式配置并不是依赖单一偶然参数点支撑起来的。

## Best-by-AUC Summary

```text
       hparam   dataset  default_value  best_value_by_auc  default_auc  best_auc  auc_gain_vs_default  default_acc  best_acc  acc_gain_vs_default  default_rmse  best_rmse  rmse_delta_vs_default
      dropout assist_09         0.3000             0.2000       0.7894    0.7922               0.0028       0.7471    0.7478               0.0007        0.4544     0.4600                 0.0055
      dropout assist_17         0.3000             0.3000       0.8005    0.8005               0.0000       0.7288    0.7288               0.0000        0.4247     0.4247                 0.0000
      dropout     junyi         0.3000             0.2000       0.8125    0.8143               0.0018       0.7709    0.7746               0.0037        0.3983     0.3966                -0.0017
embedding_dim assist_09       256.0000           384.0000       0.7891    0.7903               0.0012       0.7487    0.7470              -0.0016        0.4538     0.4563                 0.0024
embedding_dim assist_17       256.0000           256.0000       0.8006    0.8006               0.0000       0.7289    0.7289               0.0000        0.4248     0.4248                 0.0000
embedding_dim     junyi       256.0000           384.0000       0.8137    0.8141               0.0005       0.7707    0.7729               0.0022        0.3989     0.3973                -0.0016
 ortho_weight assist_09         0.5000             0.3000       0.7887    0.7893               0.0006       0.7483    0.7487               0.0003        0.4541     0.4537                -0.0004
 ortho_weight assist_17         0.5000             1.0000       0.8008    0.8008               0.0001       0.7296    0.7295              -0.0001        0.4247     0.4247                -0.0000
 ortho_weight     junyi         0.5000             0.7000       0.8136    0.8137               0.0001       0.7706    0.7696              -0.0010        0.3988     0.3989                 0.0000
```

## Recommendation

- 如果只允许优先调一个参数，建议先调 `dropout`，因为它在当前 sweep 中带来的平均 AUC 增益最高。
- 如果目标是保持当前主实验口径稳定，默认配置整体已经足够稳，不建议仅凭单因子 sweep 直接改动全部正式实验配置。
- `ACC` 与 `AUC` 并不总是同向提升，因此若要替换正式配置，建议以联合调参或二次验证为准，而不是只看单个最佳 AUC 点。
- 更适合把这组实验写成“稳定性与调参空间有限”的证据，而不是写成“大幅调参后性能显著提升”。

## Reading Guide

- 主图同时展示 `AUC` 与 `ACC`。
- 虚线表示当前默认配置位置。
- 星号标出该数据集在当前超参上的最佳 AUC 点。
- `best_vs_default_gain_summary.png` 用于展示默认配置到最佳点的净增益，更适合论文正文快速引用。
- `RMSE` 收录在 summary CSV 中，供正文或附录写作使用。