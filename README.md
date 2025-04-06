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
| Model               | Quantitative Data   | Categorical Data   | How to Handle Categorical            | Advantages                             | Disadvantages                                       |
|:--------------------|:--------------------|:-------------------|:-------------------------------------|:---------------------------------------|:----------------------------------------------------|
| Linear Regression   | ✓                   | ❌                 | One-Hot Encoding                     | Simple, fast, interpretable            | Needs scaled data, can't handle categories directly |
| Logistic Regression | ✓                   | ❌                 | One-Hot Encoding                     | Probabilistic, interpretable           | Same as above                                       |
| KNN                 | ✓                   | ❌                 | One-Hot or Ordinal                   | No training phase, simple logic        | Sensitive to scaling and distance distortions       |
| Decision Trees      | ✓                   | ✓                  | Raw labels or Ordinal                | Handles mixed data well                | Can overfit                                         |
| Random Forest       | ✓                   | ✓                  | Raw labels or Ordinal                | Robust, handles noise                  | Slower with many trees                              |
| XGBoost / LightGBM  | ✓                   | ✓                  | Label Encoding or raw (if supported) | Fast, accurate, missing-value tolerant | Encoding still affects performance                  |
| SVM                 | ✓                   | ❌                 | One-Hot Encoding                     | Good in high dimensions                | Slow on large data, needs scaling                   |
| Neural Nets         | ✓                   | ❌                 | Embeddings or One-Hot                | Flexible, powerful                     | Needs more data, complex tuning                     |
| Naive Bayes         | ✓                   | ✓                  | Label Encoding / raw                 | Handles categorical features well      | Strong independence assumption                      |
