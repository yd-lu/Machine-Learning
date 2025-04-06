# Machine-Learning

## Table of Contents
- [How to deal with missing values](#How-to-deal-with-missing-values)
- [Data Analysis Techniques](#data-analysis-techniques)
- [Python Code Examples](#python-code-examples)


### How to deal with missing values
You're aiming to keep the data as representative and unbiased as possible. The method chosen should not change the data distribution too much.
+ Methods : ffill, bfill, fill by mean, fill by median, dropna
+ Time series : prioritize ffill. 
+ < 5% Missing : probably safe to fill or drop rows. >30%	Often better to drop the column (if not essential).

### Which type of model with which type of data

import pandas as pd

data = {
    "Model": [
        "Linear Regression", "Logistic Regression", "KNN", "Decision Trees",
        "Random Forest", "XGBoost / LightGBM", "SVM", "Neural Nets", "Naive Bayes"
    ],
    "Quantitative Data": ["✓"] * 9,
    "Categorical Data": ["❌", "❌", "❌", "✓", "✓", "✓", "❌", "❌", "✓"],
    "How to Handle Categorical": [
        "One-Hot Encoding",
        "One-Hot Encoding",
        "One-Hot or Ordinal",
        "Raw labels or Ordinal",
        "Raw labels or Ordinal",
        "Label Encoding or raw (if supported)",
        "One-Hot Encoding",
        "Embeddings or One-Hot",
        "Label Encoding / raw"
    ],
    "Advantages": [
        "Simple, fast, interpretable",
        "Probabilistic, interpretable",
        "No training phase, simple logic",
        "Handles mixed data well",
        "Robust, handles noise",
        "Fast, accurate, missing-value tolerant",
        "Good in high dimensions",
        "Flexible, powerful",
        "Handles categorical features well"
    ],
    "Disadvantages": [
        "Needs scaled data, can't handle categories directly",
        "Same as above",
        "Sensitive to scaling and distance distortions",
        "Can overfit",
        "Slower with many trees",
        "Encoding still affects performance",
        "Slow on large data, needs scaling",
        "Needs more data, complex tuning",
        "Strong independence assumption"
    ]
}

model_feature_df = pd.DataFrame(data)
print(model_feature_df.to_markdown(index=False))

