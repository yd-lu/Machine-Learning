# Machine-Learning

## Table of Contents
- [How to deal with missing values](#How-to-deal-with-missing-values)
- [Which type of model with which type of data](#Which-type-of-model-with-which-type-of-data)
- [One-Hot and Label Encoding](#One-Hot-and-Label-Encoding)
- [p >> n](#p->>-n)
- [Gradient Boosting](#Gradient-Boosting)
- [How to deal with overfitting linear regression](#How-to-deal-with-overfitting-linear-regression)



## How to deal with missing values
You're aiming to keep the data as representative and unbiased as possible. The method chosen should not change the data distribution too much.
+ Methods : ffill, bfill, fill by mean, fill by median, dropna
+ Time series : prioritize ffill. 
+
| % Missing | What You Might Do                                                                 |
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




## One-Hot and Label Encoding

🔵 One-Hot Encoding

✅ drop_first=True avoids multicollinearity (for linear models).

```
import pandas as pd

df = pd.DataFrame({'Color': ['Red', 'Green', 'Blue', 'Green', 'Red']})
df_onehot = pd.get_dummies(df, columns=['Color'], drop_first=True)

print(df_onehot)
```

or

```
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse=False)
encoded = encoder.fit_transform(df[['Color']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

print(encoded_df)
```

🔵 Label Encoding

🚨 Be careful : LabelEncoder assigns arbitrary numbers, so for linear/KNN models it can introduce fake orderings like: Green < Red < Blue — which can mess up performance. Should be avoided when the notion of "distance" between points is important for the model.

```
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({'Color': ['Red', 'Green', 'Blue', 'Green', 'Red']})
le = LabelEncoder()
df['Color_encoded'] = le.fit_transform(df['Color'])

print(df)
```




## p >> n


### 🚨 Why High Dimensionality (p >> n) Is a Problem

| Problem              | What It Means                                                              |
|----------------------|----------------------------------------------------------------------------|
| Overfitting          | Model learns noise instead of signal                                       |
| Multicollinearity    | Features are linearly dependent, especially after One-Hot                  |
| Computational cost   | Training becomes slow, models can crash                                    |
| Poor generalization  | Good training accuracy but bad test accuracy                               |
| Singular matrices    | Matrix inversion fails in linear regression due to too many features       |



### 🎯 Model Handling When p >> n

| Model              | Issue with p >> n           | What to Do                                  | Affects Which Data                | Advantages                          | Disadvantages                        |
|--------------------|-----------------------------|---------------------------------------------|-----------------------------------|--------------------------------------|--------------------------------------|
| Linear Regression  | Overfitting, unstable        | Use **Ridge** (L2) or **Lasso** (L1)         | Categorical (One-Hot), All        | Regularizes, prevents overfitting   | Lasso can drop useful features       |
| Logistic Regression| Same as above                | Regularized Logistic (L1/L2)                | Same                               | Better classification boundaries    | Same issues                          |
| KNN                | Curse of dimensionality      | **Feature selection** or **PCA**            | All features                       | Reduces noise                       | PCA hard to interpret                |
| SVM                | Slower, overfit              | Kernel SVM with **regularization**          | All                                | Can handle high-dim if tuned        | Expensive in high dimensions         |
| XGBoost / LightGBM | Handles p >> n better        | **Feature selection**, Label Encoding       | High-cardinality categoricals      | Handles irrelevant features well    | Slower training                      |
| Random Forest      | Same as above                | Label Encoding or feature selection         | Categorical                        | Built-in feature importance         | Memory heavy                         |
| Neural Nets        | Overfitting, long training   | Use **Dropout**, **Regularization**, Embeddings | One-Hot categories            | Powerful and flexible               | Needs lots of data                   |
| Naive Bayes        | OK with high p               | Drop irrelevant features                    | Text / Categorical                 | Fast and simple                     | Assumes independence                 |


### ⚙️ Techniques to Deal with High-Dimensional Data

| Technique        | Description                               | Best For         | Pros                           | Cons                           |
|------------------|-------------------------------------------|------------------|--------------------------------|--------------------------------|
| Lasso (L1)       | Shrinks and sets some coefficients to 0   | Linear models    | Performs feature selection     | May drop useful features       |
| Ridge (L2)       | Shrinks coefficients                      | Linear models    | Keeps all features             | No sparsity                    |
| ElasticNet       | Combines L1 and L2                        | Linear models    | Best of both worlds            | Needs tuning                   |
| PCA              | Reduces dimensions using projection       | KNN, SVM         | Reduces dimensionality         | Less interpretable             |
| SelectKBest      | Keeps top-k important features            | Any model        | Fast, easy to use              | Might drop subtle features     |
| VarianceThreshold| Drops low-variance features               | Any model        | Simple                         | Might remove useful features   |
| Embeddings       | Dense vectors for categories              | Deep learning    | Compact and powerful           | Needs training                 |
| Hashing Trick    | Hashes categories to fixed size           | Text, Categorical| Very memory-efficient          | Risk of collisions             |



## Gradient Boosting
### Principles
Gradient Boosting is an ensemble learning method. It is a boosting algorithm which combine multiple weak learners to create a strong predictive model. In gradient boosting each new model is trained to minimize the loss function such as mean squared error of the previous model using gradient descent. In each iteration the algorithm computes the gradient of the loss function with respect to the predictions and then trains a new weak model to minimize this gradient. **If Loss = MSE, then gradient = -2(residul)**

After each tree is trained its predictions are shrunk by multiplying them with the learning rate η (which ranges from 0 to 1). This prevents overfitting by ensuring each tree has a smaller impact on the final model.

Once all trees are trained predictions are made by summing the contributions of all the trees. The final prediction is given by the formula:
$$y_{pred} = y_0(initial prediction) + \eta(r_1+...r_N)$$
where $r_i$ are the residuals (errors) predicted by each tree.


## How to deal with overfitting linear regression

Overfitting in linear regression means your model is “learning” noise rather than the true underlying relationship. Here’s how to tackle it head‑on:

- Diagnose with Cross‑Validation
  – Split your data (e.g. k‑fold CV) and monitor training vs. validation error. A big gap ⇒ overfitting.
  – Use learning curves (plot error vs. training set size) to see if more data would help.

Regularize
  – Ridge regression (L₂ penalty) shrinks coefficients toward zero
  – Lasso (L₁ penalty) can drive some coefficients exactly to zero (feature selection)
  – Elastic Net blends both L₁ and L₂—tune the mix to balance shrinkage vs. sparsity.

Simplify Your Feature Set
  – Remove weak or highly correlated predictors. Too many features relative to samples invites noise‑fitting.
  – Dimensionality reduction (e.g., PCA) to capture most variance in fewer components.

