![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Metric Uncertainty and Model Comparison

## Overview

Picking the "best" model by looking at a single accuracy number is one of the most common mistakes in applied machine learning. A model that scores 0.83 and another that scores 0.81 might not be meaningfully different once you account for sampling variability — or the first model might only look better because you evaluated dozens of candidates on the same test set and picked the winner.

In this lab you will train two or three simple scikit-learn classifiers on the same dataset and then rigorously compare them. Instead of trusting point estimates, you will compute bootstrap confidence intervals for accuracy, F1, and AUC, then run a paired permutation test to check whether the observed performance gap is statistically significant. You will also see first-hand how repeated test-set evaluation inflates the apparent performance of the chosen model — the model selection bias problem — and learn a simple mitigation strategy.

The lab ends with a practical deliverable: a structured evaluation memo that presents your model comparison, statistical evidence, and a recommendation. This is the kind of document a data scientist writes for a project stakeholder who needs to decide which model goes into production.

## Learning Goals

By the end of this lab, you should be able to:

- Train and evaluate multiple classifiers using a consistent train/test pipeline.
- Compute bootstrap confidence intervals for classification metrics (accuracy, F1, AUC).
- Run a paired permutation test to determine whether two models perform significantly differently.
- Explain model selection bias and demonstrate it experimentally.
- Apply threshold analysis to explore the precision-recall trade-off for a chosen model.
- Write a structured evaluation memo that supports a model recommendation with statistical evidence.

## Setup and Context

You will work with a binary classification dataset provided in the starter notebook. The dataset is intentionally chosen so that simple models perform reasonably well and the performance gap between them is small — exactly the scenario where statistical comparison matters most.

All modelling uses scikit-learn. The statistical testing is done with NumPy and SciPy — no specialized ML-testing libraries are needed.

## Requirements

- Fork this repo and clone it to your local machine.
- Open the notebook `m3-12-metric-uncertainty-model-comparison.ipynb`.
- Make sure the following Python packages are available (install any that are missing):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Getting Started

1. Fork and clone the repository.
2. Open the notebook and run the setup cell to load the dataset and verify your package versions.
3. Read through all six tasks before you start — the evaluation memo in Task 6 synthesizes results from every earlier task, so knowing what you need to report will help you collect the right outputs along the way.
4. Use a fixed random seed (`random_state=42`) wherever randomness is involved so your results are reproducible.

## Tasks

### Task 1: Train Baseline Models

Train two or three classifiers on the provided training set. Keep the models simple — the goal is comparison methodology, not model tuning.

1. Choose at least two of the following: Logistic Regression, Decision Tree, Random Forest, k-Nearest Neighbors.
2. Use default hyperparameters (or minimal tuning) so the focus stays on evaluation.
3. Generate predictions and predicted probabilities (`predict_proba`) on the held-out test set.
4. Print a classification report and the ROC-AUC score for each model.

Deliverable: trained models, test-set predictions, and a printed comparison table showing accuracy, F1, and AUC for each model.

### Task 2: Bootstrap Confidence Intervals

For each model, compute 95 % bootstrap confidence intervals for accuracy, F1-score, and AUC:

1. Write a reusable function `bootstrap_metric(y_true, y_pred_or_proba, metric_fn, n_boot=2000, seed=42)` that returns the point estimate and the 2.5th/97.5th percentile interval.
2. Apply the function to each model and each metric.
3. Visualize the results: create a grouped bar chart (or dot-and-whisker plot) where each model has bars for accuracy, F1, and AUC, with error bars showing the 95 % CI.
4. In a Markdown cell, answer: **Do any of the confidence intervals overlap? What does that tell you about whether the models are meaningfully different?**

Deliverable: a reusable bootstrap function, a CI comparison plot, and a written interpretation.

### Task 3: Paired Permutation Test

Use a paired permutation test to formally compare the two best-performing models:

1. For each test-set sample, record whether each model predicted correctly (1) or not (0). This gives you a paired vector of differences.
2. Implement (or use `scipy.stats.permutation_test`) a two-sided permutation test on the paired accuracy differences.
3. Report the observed difference in accuracy, the permutation p-value, and a plain-language conclusion at the α = 0.05 level.
4. Repeat the test for F1-score: convert per-sample predictions into per-sample F1 contributions using the micro-averaging trick, or test at the aggregate level by bootstrapping the F1 difference.

Deliverable: permutation test results for accuracy and F1, with printed p-values and a written conclusion.

### Task 4: Model Selection Bias Demonstration

Show how evaluating many models on the same test set inflates the apparent performance of the winner:

1. Train 20 Decision Tree classifiers with different `max_depth` values (e.g., 1 through 20).
2. Evaluate all 20 on the same test set and record the best test-set accuracy.
3. Now estimate the "honest" performance of that best model using 5-fold cross-validation on the training set only.
4. Compare the two numbers. In a Markdown cell, explain:
   - Why the best test-set accuracy is likely an overestimate.
   - How cross-validation or a separate validation set mitigates this bias.
   - When this bias matters most in practice (hint: small test sets, many candidates).

Deliverable: a table or plot of test-set accuracy vs. `max_depth`, the cross-validated estimate of the chosen model, and a written explanation of model selection bias.

### Task 5: Threshold Analysis

Pick the model you would recommend and explore how changing the classification threshold affects precision and recall:

1. Plot a precision-recall curve using `sklearn.metrics.PrecisionRecallCurve`.
2. On the same figure, mark three thresholds: the default 0.5, a threshold that maximizes F1, and a threshold that achieves at least 90 % precision (if reachable).
3. For each threshold, print the corresponding precision, recall, and F1.
4. In a Markdown cell, write 2–3 sentences advising a stakeholder: **If false positives are twice as costly as false negatives, which threshold would you recommend and why?**

Deliverable: an annotated precision-recall curve, a threshold comparison table, and a stakeholder-facing recommendation.

### Task 6: Evaluation Memo

Write a structured Markdown memo (directly in your notebook) that a project lead could read to decide which model to deploy. Use the following template:

```
## Evaluation Memo

### Objective
(One sentence: what decision does this memo support?)

### Models Compared
(Table with model names, key hyperparameters, and point-estimate metrics.)

### Statistical Comparison
(Bootstrap CIs and permutation test results. State whether the
performance difference is statistically significant.)

### Model Selection Bias Check
(Summarize Task 4 findings. State the honest performance estimate.)

### Threshold Recommendation
(Summarize Task 5. State the recommended threshold and the
expected precision/recall at that threshold.)

### Recommendation
(Which model, at which threshold, and why — in 3–5 sentences.)
```

Deliverable: a completed evaluation memo in Markdown format.

## Submission

### What to submit

- Your completed Jupyter notebook (`m3-12-metric-uncertainty-model-comparison.ipynb`) with all tasks, outputs, and Markdown responses.

### Definition of done

- [ ] Two or three classifiers trained and evaluated with classification reports (Task 1).
- [ ] Bootstrap 95 % CIs computed for accuracy, F1, and AUC with a comparison visualization (Task 2).
- [ ] Paired permutation test results with p-values and written conclusions (Task 3).
- [ ] Model selection bias demonstrated with a comparison of test-set best vs. cross-validated estimate (Task 4).
- [ ] Precision-recall curve with three annotated thresholds and a stakeholder recommendation (Task 5).
- [ ] Completed evaluation memo with all sections filled in (Task 6).
- [ ] All cells run without errors from top to bottom.

### How to submit

```bash
git add .
git commit -m "lab: complete metric uncertainty and model comparison"
git push origin main
```

Upload your file to the learning platform when finished.
