# COMP472-Project

COMP 472, Concordia University, Fall 2026, Kaggle project. We wrote k-NN, a decision tree, a random forest and a
feed-forward neural network, and compared them on two sets of pre-extracted feature vectors: image, with 500 features
and 50 classes, and sentiment, with 768 features and 3 classes. We also ran a k-means clustering experiment.

Final submission file is `submissions/2026-11-16_knn+nn0.2-balanced_nn-rebalance.csv` with public score of 0.662. Image uses
0.8 × k-NN + 0.2 × neural network and sentiment uses a neural network rebalanced to 1/3 per class.

## How to run

Needs Python 3.13 and the five Kaggle CSVsv in the data folder. Run everything from the repository root.

```bash
# 1. environment
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. data: put image_train.csv, image_test.csv, sentiment_train.csv, sentiment_test.csv
#    and sample_submission.csv in data/

# 3. retrain the final models and rebuild the submission (about 10 s, prints nothing)
python -m experiments.final_retrain --dataset image --model knn --run k100_cos_exp0.03_logtfidf --overwrite
python -m experiments.final_retrain --dataset image --model nn --run h256_lr1e-3_do0.2_wd0_adam_gelu_bs256_standardise --epochs 5 --overwrite
python -m experiments.final_retrain --dataset sentiment --model nn --run h256x64_lr1e-4_do0_wd0_adam_raw --epochs 16 --overwrite
python -m src.make_submission --image knn --sentiment nn --image-mix nn:0.2 --image-balance --sentiment-correction rebalance

# 4. compare with the file we submitted
python -m src.compare_submissions submissions/$(date +%F)_knn+nn0.2-balanced_nn-rebalance.csv \
                                  submissions/2026-11-16_knn+nn0.2-balanced_nn-rebalance.csv
```


## 1. Files

```
COMP472-Project/
├── requirements.txt    pinned versions of the libraries we import
├── ENVIRONMENT.txt     Python version and full pip freeze of the environment that produced the results
├── data/               competition CSVs from Kaggle, not committed
├── splits/             fixed train/validation split and 5 folds, one .npz per dataset, plus test ids
├── src/                models, data loading, logging, submission code: table below
├── experiments/        one script per experiment: 
├── runs/               one folder per trained model: config.json with settings and scores
├── results/            results.csv, one row per run, and one CSV per experiment
├── figures/            figures made by the experiments and the notebook
├── notebooks/          error analysis: per-class scores, confusion matrices
└── submissions/        every submission CSV we built
```

| `src/` file | Purpose |
|---|---|
| `config.py` | Seed 42, folder paths, number of classes per dataset |
| `seed.py` | `set_seed()` seeds `random`, NumPy and PyTorch |
| `data.py` | Reads and checks the CSVs, returns the split with a transform fitted on training rows only |
| `make_splits.py` | Made the committed split once. Refuses to overwrite it |
| `base.py` | `BaseModel`: the `fit`, `predict_proba`, `predict` interface every classifier follows |
| `knn.py` | k-NN in NumPy: three distances, three voting rules |
| `tree.py` | Decision tree in NumPy: Gini or entropy, size limits, quantile thresholds |
| `forest.py` | Random forest in NumPy, built on `tree.py`: bootstrap samples, random features per split |
| `nn.py` | Feed-forward neural network in PyTorch, with early stopping on validation loss |
| `cluster.py` | k-means in NumPy with k-means++ starts, scored with ARI and NMI |
| `prior.py` | Class-mix tools: sentiment rebalancing, equal image class counts |
| `logger.py` | The only code that writes to `runs/` and `results/`. Refuses to overwrite a run |
| `make_submission.py` | Builds and checks a submission CSV from the final test probabilities |
| `compare_submissions.py` | Counts the labels two submission files share |

## 2. Setup and data pre-processing

1. Environment and data: steps 1 and 2 of How to run. The results came from Python 3.13.6 (full list in `ENVIRONMENT.txt`).
2. Use the committed split in `splits/`, which is an 80/20 validation split with seed 42 plus 5 folds.
3. Call `load()` with a transform. It fits the transform on training rows only and applies it to validation rows.
   With `final=True` it fits on all training rows and returns the test rows.


To run the pre-processing on its own:

```bash
python -c "from src.data import load; Xtr, ytr, Xva, yva = load('image', 'logtfidf'); print(Xtr.shape, Xva.shape)"

```

## 3. Train and evaluate

### 3.1 One model

```python
from src.seed import set_seed
from src.data import load
from src.knn import KNN

set_seed()
Xtr, ytr, Xva, yva = load("image", transform="logtfidf")
knn = KNN(n_classes=50, k=100, metric="cosine", weights="exp", temperature=0.03).fit(Xtr, ytr)
print((knn.predict(Xva) == yva).mean())    # 0.597
```

`DecisionTree` in `src/tree.py`, `RandomForest` in `src/forest.py` and `MLP` in `src/nn.py` work the same way.

### 3.2 Every experiment

Each script in `experiments/` trains models, scores them on the validation split and logs them. Its curves are stored in 
`results/` and its figures to `figures/`. To rerun from scratch, move the committed outputs first:
```bash
mv runs runs_committed && mv results results_committed && mkdir results
```

Run the steps in order, because later steps read the runs that earlier steps log. Steps 1 and 3–8 take `--dataset`, so
run each of them twice: once with `--dataset image` and once with `--dataset sentiment`. 

| # | Purpose | Command | Time |
|---|---|---|---|
| 1 | k-NN: k, distance, voting rule, transform | `python -m experiments.knn_sweep --dataset image` | 10 min |
| 2 | k-NN exponential votes: k, temperature | `python -m experiments.knn_exp_sweep` | 1 min |
| 3 | First decision tree runs | `python -m experiments.tree_baseline --dataset image` | 5 min |
| 4 | Tree: depth, split and leaf sizes, criterion | `python -m experiments.tree_sweep --dataset image` | 2 h |
| 5 | Forest: trees, depth, features per split | `python -m experiments.forest_sweep --dataset image` | 14 h |
| 6 | Untuned network with loss curves | `python -m experiments.nn_baseline --dataset image` | 2 min |
| 7 | Network settings | `python -m experiments.nn_sweep --dataset image` | 10 min |
| 8 | k-means over the number of clusters | `python -m experiments.cluster_sweep --dataset image` | 10 min |
| 9 | k-means inside each image class | `python -m experiments.subcluster_sweep` | 5 min |
| 10 | Fit time, predict time and size | `python -m experiments.compute_cost` | 1 min |
| 11 | 5-fold cross-validation of the finalists | `python -m experiments.cv_finalists` | 1 h |
| 12 | Sentiment class mix | `python -m experiments.prior_correction` | 1 min |
| 13 | Final model choice | `python -m experiments.choose_final` | 1 min |
| 14 | Image k-NN + network mix and equal class counts | `python -m experiments.image_blend` | 2 min |
| 15 | Image accuracy by neighbour similarity | `python -m experiments.knn_similarity` | 1 min |



## 4. Apply the model

Steps 3 and 4 of How to run retrain the final models on all training rows and rebuild the submitted file.
`final_retrain` reads the validated settings from the run's `config.json` and writes `runs/{d}/{model}/final/`.
`make_submission` mixes and rebalances the test probabilities, matches them to `sample_submission.csv` by id, and
writes `submissions/{today}_knn+nn0.2-balanced_nn-rebalance.csv`. `compare_submissions` counts how many labels it
shares with the submitted file.
