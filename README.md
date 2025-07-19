# 🧠 Learned Token Pruning: Patterns and Performance

## 📘 Overview

This project investigates the effectiveness and impact of **Learned Token Pruning (LTP)** on transformer-based models, focusing on **efficiency**, **performance**, and **interpretability**. By pruning less informative tokens during fine-tuning, the goal is to reduce inference time while preserving classification accuracy.

We conduct experiments on short-text datasets like **AG News** and **IMDb Reviews**, analyzing token retention patterns and how pruning affects domain-specific vocabulary, performance, and semantic separability.

---

## 🎯 Objectives

- Apply and evaluate LTP on fine-tuned RoBERTa models.
- Analyze the nature of retained tokens post-pruning.
- Evaluate the impact of pruning on model accuracy and interpretability.
- Study domain-specific token retention and performance impact.

---

## 🧪 Methodology

### 🔹 Experiment 1: Model Training and Pruning
- **Baseline**: Fine-tune RoBERTa without pruning to establish benchmark metrics.
- **LTP Fine-Tuning**: Integrate the LTP mechanism into fine-tuning and dynamically prune tokens during training.
- **Threshold Adaptation**: Use soft pruning to allow layers to learn their pruning thresholds.

### 🔹 Experiment 2: Token Pattern Analysis
- **Extract Retained Tokens**: After pruning, gather all tokens that remain.
- **POS Tagging & Categorization**: Use `spaCy` to classify tokens into stopwords, adjectives, named entities, etc.
- **Frequency & Category Stats**: Compute retention frequencies per class.
- **Clustering**: Use t-SNE to visualize retained token distributions.

### 🔹 Experiment 3: Performance Evaluation
- **Metrics**: Accuracy and F1-score for baseline vs. pruned models.
- **Comparison**: Measure loss/gain in performance post-pruning.

### 🔹 Experiment 4: Domain-Specific Retention
- **Identify Keywords**: Manually select domain-specific keywords per dataset.
- **Retention Tracking**: Check if pruning retains or discards these terms.
- **Effect on Output**: Assess classification performance related to retained domain-specific vocabulary.

---

## 📊 Datasets

| Dataset     | Domain   | Description |
|-------------|----------|-------------|
| AG News     | General  | News articles in four categories: World, Sports, Business, Sci/Tech (~120 words each). |
| IMDb Reviews| Sentiment| Movie reviews labeled as positive or negative (~200 words each). |

---

## 📁 Files and Their Uses

### 🔹 Model Training Pipelines
- `model_pipeline_agnews.ipynb` — Training pipeline using AG News dataset  
- `model_pipeline_imdb.ipynb` — Training pipeline using IMDb dataset

### 🔹 Trained Models (Post Phase 3)
- `ag_news_distilbert_hard/` — Final AG News model checkpoint - [Link here](https://drive.google.com/drive/folders/1jM8vA-b6g4q5cGAWh5AxTtjn6wG2kIJ6?usp=share_link)
- `imdb_distilbert_hard/` — Final IMDb model checkpoint - [Link here](https://drive.google.com/drive/folders/1oY8q8zXdxVc5AkxZEU6qRbZ5oAzg9UYA?usp=share_link)

### 🔹 Pruned CSV Outputs
Each CSV contains:
- Initial sentence
- Pruned sentence (final retained tokens)
- True and predicted labels
- POS tags of retained sentence

Files:
- `cleaned_agnews_pruned.csv`
- `cleaned_imdb_pruned.csv`

### 🔹 Results and Visualizations
- `results_agnews/` — Graphs and token analysis for AG News
- `results_imdb/` — Graphs and token analysis for IMDb

---

## 📈 Results Summary

### 🟠 IMDb
- **Positive Sentiment**: Tokens like _“great”_, _“amazing”_, _“love”_, _“recommend”_ retained.
- **Negative Sentiment**: Words such as _“worst”_, _“awful”_, _“boring”_, _“waste”_ preserved.
- **POS Tags**: Adjectives and nouns dominate; content-bearing words are preserved more than grammatical ones.

### 🔵 AG News
- **World**: geopolitics (e.g., _“president”_, _“iraq”_, _“troops”_)
- **Sports**: temporal/action terms (_“final”_, _“season”_, _“win”_)
- **Business**: market terms (_“stocks”_, _“profit”_, _“shares”_)
- **Sci/Tech**: tech nouns (_“google”_, _“linux”_, _“space”_)

### 🔍 t-SNE Clustering
- IMDb: Clear positive vs. negative separation; denser positive clusters.
- AG News: Coherent clusters for each topic, showing strong domain-specific token retention.

---

## 📌 Key Insights

- Pruning preserves **semantically relevant, high-impact tokens**.
- POS distribution favors **adjectives, nouns, and verbs** that contribute meaning.
- Domain-specific keywords are **effectively retained**, supporting classification.
- LTP leads to significant **token reduction** with **minimal accuracy drop**.


---

## 🧠 Expected Outcomes

- **Token Pruning Patterns**: Insight into which token types are preserved vs. dropped.
- **Performance Trade-offs**: Measure LTP’s effect on classification metrics.
- **Interpretability**: Clarity on how models make decisions post-pruning.

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- spaCy
- scikit-learn
- Matplotlib, Seaborn

Install via:

```bash
pip install -r requirements.txt




