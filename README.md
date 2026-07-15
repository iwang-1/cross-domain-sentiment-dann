# Unsupervised Domain Adaptation for Cross-Domain Sentiment

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iwang-1/cross-domain-sentiment-dann/blob/main/cross_domain_sentiment.ipynb)

Small-data sentiment classification across Yelp, Amazon, Twitter (X), and Reddit
using DistilBERT and a **Domain-Adversarial Neural Network (DANN)** with a
gradient-reversal layer.

The DANN stage trains its sentiment head on labeled source-domain examples while
its domain head sees both source text and **unlabeled target-domain text**. Target
labels are withheld from the training loss and used only for evaluation. That is
unsupervised domain adaptation, not strict unseen-domain generalization.

> A cleaned-up public release of a deep-learning project originally developed in
> Google Colab. The notebook, smoke-test results, report, and generated figures
> are retained so the implementation and evidence boundary are inspectable.

## What is implemented

- Four source/target rotations across Yelp, Amazon, Twitter, and Reddit.
- Four labeled-source fractions per rotation, producing 16 training conditions.
- DistilBERT with sentiment and domain-classification heads connected through a
  gradient-reversal layer.
- Balanced source-domain sampling, text normalization, synonym augmentation,
  per-epoch logging, and accuracy/F1 evaluation.
- A t-SNE representation probe plus a separate Financial PhraseBank evaluation.

## Evidence boundary

The committed machine-readable artifact is
[`results/dann_results.csv`](results/dann_results.csv): a three-epoch,
smoke-test-scale run across all 16 source/target conditions. Its target splits are
only about 30 examples, so those accuracies are useful for checking that the
pipeline executes, not as a defensible benchmark headline.

The PDF report contains results from earlier interactive Colab runs, but the
exact sampled data, checkpoints, and run logs needed to reproduce those tables
are not committed. This README therefore does not promote the report-only
accuracy figures as verified results.

---

## Why the domains differ — a look at the representations

Before any adaptation, DistilBERT's `[CLS]` embeddings already cluster by platform:
Yelp and Amazon reviews sit close together, Twitter and Reddit overlap in their own
region, and Financial PhraseBank is off on its own. That separation *is* the domain
shift — the model encodes "which platform" as strongly as "which sentiment," which
is exactly what adversarial training tries to erase.

![t-SNE of DistilBERT CLS embeddings, colored by domain](docs/img/tsne_embeddings.png)

---

## Method

**Task.** Binary sentiment (positive / negative) on balanced samples from each
platform.

**Source/target rotation.** For each target platform, the other three platforms
provide labeled source examples. Target-platform text participates in the
domain-adversarial loss without sentiment labels; those labels are read only for
evaluation. Each rotation runs at four labeled-source fractions (10% / 25% /
50% / 100%), producing 16 conditions that probe data scarcity and domain shift.
Because target text is available during training, these are adaptation
experiments rather than strict holdout-generalization experiments.

**Domain-adaptation components.**

| Technique | What it does |
|---|---|
| Text normalization | Lowercasing, emoji→text, URL/mention/hashtag stripping, repeated-char collapse |
| Synonym augmentation | WordNet synonym replacement (~10% of tokens) to diversify limited data |
| Balanced domain sampling | `WeightedRandomSampler` for equal per-domain representation each batch |
| **DANN** | Gradient-reversal layer + adversarial domain-classification head on the shared DistilBERT encoder, pushing it toward domain-invariant features |

**DANN architecture.** A shared DistilBERT encoder feeds two heads: a sentiment
classifier and a domain classifier. A gradient-reversal layer sits before the domain
head, so minimizing domain loss *maximizes* domain confusion in the encoder —
encouraging representations that transfer across platforms.

**Smoke-run training curves** (per target domain and labeled-source fraction):

| Loss | Accuracy |
|---|---|
| ![DANN loss curves](docs/img/dann_loss_curves.png) | ![DANN accuracy curves](docs/img/dann_accuracy_curves.png) |

---

## Repository layout

```
cross_domain_sentiment.ipynb   Data prep, source/target splits, DANN, t-SNE, plots
results/dann_results.csv       Per-epoch logs for the committed smoke-scale run
report/final_report.pdf        Earlier write-up; tables are not reproducible artifacts
report/proposal.pdf            Original project proposal
docs/img/                      Figures used in this README
scripts/make_results_chart.py  Visualizes report-only numbers (not benchmark evidence)
requirements.txt               Python dependencies
```

## Reproducing

The notebook was developed in **Google Colab** (GPU recommended) and expects a few
external inputs, so it is not a one-command local run. To reproduce:

1. Open `cross_domain_sentiment.ipynb` in Colab and `!pip install emoji`.
2. Download NLTK data for synonym augmentation:
   ```python
   import nltk; nltk.download("wordnet"); nltk.download("omw-1.4")
   ```
3. Datasets download automatically via `kagglehub` (Yelp; Twitter/Reddit — needs a
   free Kaggle login) and 🤗 `datasets` (Amazon `mteb/amazon_polarity`; Financial
   PhraseBank `financial_phrasebank`, `sentences_75agree`).
4. Run cells top to bottom: data prep → source/target splits → DANN → evaluation
   → plots. Dataset versions and sampling environment affect exact outputs; use
   the committed CSV to inspect the retained smoke-scale run.

## Datasets

| Domain | Source |
|---|---|
| Yelp | Kaggle `yelp-dataset/yelp-dataset` |
| Amazon | 🤗 `mteb/amazon_polarity` |
| Twitter / Reddit | Kaggle `cosmos98/twitter-and-reddit-sentimental-analysis-dataset` |
| Financial PhraseBank (OOD) | 🤗 `financial_phrasebank` (`sentences_75agree`) |

Raw data is **not** committed — it is fetched at runtime from the sources above.

## Authors

Ivan Wang, Evan Zhang, Unlam Leong, Jessie Gu, Raymond Chen.

## License

Code released under the [MIT License](LICENSE). The PDF write-up and proposal in
`report/` are the authors' work, included for reference.
