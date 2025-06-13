# SMS Spam Detection with TF-IDF & Multinomial Naïve Bayes

A compact, **fully-reproducible Jupyter notebook** (<kbd>spam_detection_naive_bayes.ipynb</kbd>) that trains and evaluates a text-classification model on the classic SMS Spam Collection dataset.  
In ~60 lines of Python we achieve **≈ 96 % accuracy** and an **AUC ≈ 0.99**—all on a single CPU in seconds.

---

## Table of Contents
1. [Features](#features)  
2. [Quick Start](#quick-start)  
3. [Project Structure](#project-structure)  
4. [Notebook Walk-through](#notebook-walk-through)  
5. [Results Snapshot](#results-snapshot)  
6. [Next Steps](#next-steps)  
7. [License](#license)

---

## Features
- **End-to-end workflow**: data download → preprocessing → train/test split → TF-IDF vectorisation → model fitting → evaluation plots.  
- **Lightweight model**: Multinomial Naïve Bayes pairs naturally with sparse TF-IDF features.  
- **Interpretability hooks**: view top “spammy” tokens by log-odds.  
- **Diagnostic visuals**: confusion matrix, ROC curve, precision–recall curve.

---

## Quick Start

### 1 · Clone and install
```
git clone https://github.com/&lt;your-user&gt;/sms-spam-nb.git
cd sms-spam-nb
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```


### 2 · Run the notebook
```
jupyter notebook spam_detection_naive_bayes.ipynb
```

> **Tip:** No GPU required, executes in under a minute on most laptops.

---

## Project Structure
```
spam_detection_naive_bayes/
│
├─ spam_detection_naive_bayes.ipynb   ← Main notebook (step-by-step tutorial)
└─ README.md                          ← You are here
```

---

## Notebook Walk-through

| Section | What happens |
|---------|--------------|
| **1 · Fetch the dataset** | Downloads the SMS Spam Collection via `kagglehub`. |
| **2 · Inspect the raw data** | Loads CSV, shows class balance, cleans column names. |
| **3 · Split & vectorise** | 80/20 train/test split, converts text to TF-IDF vectors, removes English stop-words. |
| **4 · Model training** | Fits `MultinomialNB()` in milliseconds. |
| **5 · First-pass metrics** | Accuracy, precision, recall, F1—spotlight on spam-recall gap. |
| **6–8 · Visual diagnostics** | Confusion matrix heatmap, ROC curve (AUC), Precision-Recall curve (AP). |
| **9 · “Spammy” words** | Lists highest log-odds tokens (`free`, `txt`, `win`, …). |
| **10 · Sanity-check messages** | Runs handcrafted examples through the pipeline. |
| **Conclusion** | Recaps strengths & limitations; outlines tuning ideas. |

---

## Results Snapshot
| Metric | Score |
|--------|-------|
| **Accuracy** | 96.7 % |
| **AUC - ROC** | 0.99 |
| **Ham Recall** | 1.00 |
| **Spam Recall** | 0.75 |

🔎 **Interpretation:** No legit messages are mis-flagged (ham recall = 1.00), but 25 % of spam slips past—common for out-of-the-box NB. See notebook for tuning suggestions.

---

## Next Steps
* **Boost spam recall**: class-prior tweaks, add bi-/tri-grams, or switch to linear SVM / logistic regression.  
* **Deploy**: wrap the vectoriser + model in a Flask/FastAPI microservice.  
* **Explainability**: integrate SHAP for per-message feature attributions.  
* **Dataset augmentation**: blend newer SMS/phishing corpora to stay current.

