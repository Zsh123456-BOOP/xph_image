# Run R Plotting Scripts

Use the commands below to generate publication-quality plots.
All outputs are saved to `R_image/exp{N}/{dataset}/` as individual PDF files (no titles).

## 1. Experiment 1 (Disentanglement)
**Datasets:** assist_09, assist_17, junyi (Total 18 main PDFs + split subplots)
```bash
Rscript /home/zsh/xph_image/R_image/plot_exp1.R
```

## 2. Experiment 2 (Gating Consistency)
**Datasets:** assist_09, assist_17, junyi (Total 6 main PDFs + split subplots)
```bash
Rscript /home/zsh/xph_image/R_image/plot_exp2.R
```

## 3. Experiment 3 (Interaction & Q-Noise)
**Dataset:** assist_09 only (Total 9 PDF files including split combos)
```bash
Rscript /home/zsh/xph_image/R_image/plot_exp3.R
```

## Run All Sequence
```bash
Rscript /home/zsh/xph_image/R_image/plot_exp1.R && \
Rscript /home/zsh/xph_image/R_image/plot_exp2.R && \
Rscript /home/zsh/xph_image/R_image/plot_exp3.R
```
