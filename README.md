# Machine-Learning

## Table of Contents
- [How to deal with missing values](#How-to-deal-with-missing-values)
- [Which type of model with which type of data](#Which-type-of-model-with-which-type-of-data)
- [Python Code Examples](#python-code-examples)



## How to deal with missing values
You're aiming to keep the data as representative and unbiased as possible. The method chosen should not change the data distribution too much.
+ Methods : ffill, bfill, fill by mean, fill by median, dropna
+ Time series : prioritize ffill. 
+ | % Missing | What You Might Do                                                                 |
|-----------|------------------------------------------------------------------------------------|
| < 5%      | Probably safe to fill or drop rows                                                 |
| 5–15%     | Imputation usually fine, especially if column is important                         |
| 15–30%    | Think carefully. Maybe keep, maybe drop. Use your domain knowledge.                |
| >30%      | Often better to drop the column (if not essential), or replace it with a model prediction |
| >50%      | Usually not worth saving unless the feature is very important and imputation is smart |



## Which type of model with which type of data
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


| Method           | When to Use                    | Description                           | Good For         | Not Good For         |
|:-----------------|:-------------------------------|:--------------------------------------|:-----------------|:---------------------|
| One-Hot Encoding | Few categories                 | Binary column per category            | Linear, KNN, SVM | High-cardinality     |
| Label Encoding   | Tree-based models              | Map categories to integers            | XGBoost, RF      | Linear models        |
| Ordinal Encoding | Ordered categories             | Map with order (e.g., low/med/high)   | Ordinal features | Unordered categories |
| Target Encoding  | High-cardinality, with caution | Encode using mean target per category | Power users      | Small datasets       |
| Embeddings       | Neural networks                | Learn dense vector per category       | Deep models      | Simple models        |
