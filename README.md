# Indic Climate Stance Challenge

Stance detection in Hindi and Bengali social-media-style text (NLP course project, Milestone 2).

## Task
Given a text, predict its stance toward one fixed claim: **"Climate change and global warming is a serious concern."**

| Label | Meaning |
|---|---|
| Favour | Agrees with or supports the claim |
| Against | Denies, mocks or downplays the claim (including sarcasm) |
| None | Neutral, unrelated, or no clear stance |

Primary metric: **Macro F1** (plus per-class F1, accuracy, per-language scores).

## Data
| Source | Language | Size | Notes |
|---|---|---|---|
| [Dhoroni](https://huggingface.co/datasets/ciol-research/Dhoroni) (MIT licence) | Bengali | 2,300 news articles | Stance annotation mapped to Favour / Against / None: [describe your mapping] |
| `data/hindi.csv` (self-written, LLM-assisted) | Hindi / Hinglish | 120 rows (40 per class) | Synthetic; Devanagari, Roman script, code-mixed, negation, sarcasm. Small and cleaner than real tweets |

Stratified split 70/15/15 by language and label (seed 42). Test set: 363 rows (Favour 260, None 76, Against 27).

## Method
1. **Baseline:** TF-IDF (word 1-2 grams + char 2-5 grams) + Logistic Regression
2. **Fine-tuned transformers:** MuRIL (`google/muril-base-cased`) and XLM-R (`xlm-roberta-base`); class-weighted cross-entropy, AdamW (lr 2e-5), batch size 16, max length 128, 4 epochs, mixed precision; best epoch chosen on validation Macro F1 only (no test leakage)
3. **Ensemble:** average of MuRIL and XLM-R class probabilities
4. **Analysis:** confusion matrices, per-language scores, negation / code-mixing slices, error analysis

Not yet implemented (planned for the final evaluation): noise-aware preprocessing, augmentation (back-translation, spelling noise), augmentation ablation.

## Results (test set)
| Model | Macro F1 | Accuracy | F1 Against | F1 Favour | F1 None | Macro F1 bn | Macro F1 hi |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.429 | n/a | n/a | n/a | n/a | n/a | n/a |
| MuRIL | 0.465 | 0.697 | 0.233 | 0.821 | 0.341 | 0.398 | 0.152 |
| XLM-R | 0.510 | 0.689 | 0.267 | 0.801 | 0.462 | 0.414 | 0.514 |
| Ensemble | **0.514** | 0.697 | 0.279 | 0.808 | 0.454 | 0.413 | 0.514 |

Key findings: transformers beat TF-IDF; the ensemble is only marginally above XLM-R; **Against** is the hardest class (class imbalance); many errors are short Bengali news headlines with implicit stance. Hindi and style slices have very few test rows (about 5-18) and are indicative only. Full outputs are in `results/`.

## Repository structure
```
├── README.md
├── requirements.txt
├── data/
│   └── hindi.csv
├── notebooks/
│   └── indic_climate_stance.ipynb     # full pipeline: data, EDA, baseline, fine-tuning, evaluation
├── results/
│   ├── results.csv
│   ├── errors.csv
│   └── confusion_*.png
├── figures/
│   ├── dataset_stats.png
│   └── baseline_confusion.png
└── report/
    └── progress_report.pdf
```

## How to run
1. Open `notebooks/indic_climate_stance.ipynb` in Google Colab and select **Runtime > Change runtime type > T4 GPU**.
2. Upload `data/hindi.csv` to the Colab Files panel.
3. Run all cells in order. Part A builds the dataset and TF-IDF baseline and writes `train.csv`, `val.csv`, `test.csv`; Part B fine-tunes MuRIL and XLM-R and writes everything in `results/`.
4. Training takes about 2 minutes per model on a T4 with this data size.

## Team contributions
| Member | Contribution |
[Tripti Priya]-MuRIL and XLM-R fine-tuning and ensemble
Models and results table
[Saachi Suri]-Evaluation, error analysis, report and GitHub repository
Error analysis, report, README
[Vrinda]-Hindi/Hinglish dataset creation and cleaning
hindi.csv (120 examples)
[Shravan]-Bengali data: Dhoroni loading and label mapping
Bengali part of the dataset
[Piyush Dagar]-Dataset statistics, charts and TF-IDF baseline
Statistics, class charts, baseline Macro F1 0.429

## Limitations
- Hindi data is small and synthetic; Hindi scores are unreliable
- Strong class imbalance (few Against examples)
- Bengali data is news text rather than social media
- Single run per model (no multi-seed averages yet)

## Future work
More real Hindi/Bengali social-media data, augmentation and noise-aware preprocessing, BanglaBERT and IndicBERT, LoRA, multi-seed runs, deeper error analysis on implicit stance and sarcasm.

## References
- Wasi et al., *Dhoroni: Exploring Bengali Climate Change and Environmental Views with a Multi-Perspective News Dataset and NLP*, arXiv:2410.17225 (dataset DOI 10.5281/zenodo.13695110)
- Khanuja et al., MuRIL; Conneau et al., XLM-R
- Mohammad et al., SemEval-2016 Task 6 (stance detection)
