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

Summary Table: Model × Feature Type × Preprocessing
Model	Quantitative Data	Categorical Data	How to Handle Categorical	Advantages	Disadvantages
Linear Regression	✓	❌	Must encode (e.g., One-Hot Encoding)	Simple, fast, interpretable	Can’t handle categories natively; needs scaled data
Logistic Regression	✓	❌	Must encode (e.g., One-Hot)	Probabilistic, interpretable	Same as above
KNN	✓	❌	Encode with One-Hot or Ordinal	Non-parametric, no training phase	Sensitive to feature scaling; categorical encoding can mislead distances
Decision Trees	✓	✓	Can take raw labels or Ordinal directly	Handles mixed data well	Can overfit
Random Forest	✓	✓	Can take raw labels or Ordinal directly	Robust, less tuning	Slower with many trees
XGBoost / LightGBM	✓	✓	Can handle raw categorical labels if specified; else use Label Encoding	Fast, accurate, handles missing values	Some preprocessing still helps; encoding matters
SVM	✓	❌	Must encode (e.g., One-Hot)	Good in high dimensions	Not ideal for large datasets; bad with raw categories
Neural Nets	✓	❌	Encode using Embeddings or One-Hot	Flexible, powerful	Needs more data/tuning
Naive Bayes	✓	✓	Use Label Encoding or leave as string (in text models)	Works well with categorical	Assumes independence

