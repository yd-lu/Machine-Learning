# Machine-Learning

## Table of Contents
- [How to deal with missing values](#How-to-deal-with-missing-values)
- [Which type of model with which type of data](#Which-type-of-model-with-which-type-of-data)
- [One-Hot and Label Encoding](#One-Hot-and-Label-Encoding)
- [p >> n](#p->>-n)
- [Gradient Boosting](#Gradient-Boosting)
- [How to deal with overfitting linear regression](#How-to-deal-with-overfitting-linear-regression)
- [t-test and F-test](t-test-and-F-test)
- [Homoscedasticity and heteroscedasticity](Homoscedasticity-and-heteroscedasticity)
- [Random Forest feature importances](Random-Forest-feature-importances)



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

- Regularize
  – Ridge regression (L₂ penalty) shrinks coefficients toward zero
  
  – Lasso (L₁ penalty) can drive some coefficients exactly to zero (feature selection)
  
  – Elastic Net blends both L₁ and L₂—tune the mix to balance shrinkage vs. sparsity.

- Simplify Your Feature Set
  
  – Remove weak or highly correlated predictors. Too many features relative to samples invites noise‑fitting.
  
  – Dimensionality reduction (e.g., PCA) to capture most variance in fewer components.



## t-test and F-test

## t‑Test for an Individual Coefficient

## Purpose  
Test whether a single predictor `x_j` has any explanatory power once you’ve controlled for the other variables.

## Null hypothesis  
- **H₀:** `β_j = 0` (no linear relationship between `x_j` and `y`, given the other regressors)  
- **H₁:** `β_j ≠ 0`

## Test statistic  
```
 t_j = beta_hat_j / SE(beta_hat_j)
 SE(beta_hat_j) = sqrt( sigma_hat^2 * (X^T X)^(-1)_[jj] )
```  
- `sigma_hat^2 = SSE/(n-p)` is the residual variance estimate, `p` = number of parameters (including intercept).  
- `(X^T X)^(-1)_[jj]` is the j-th diagonal element of `(X^T X)^(-1)`.

## Distribution and decision rule  
Under H₀,  
```
 t_j ~ t_{n-p}
```  
(degrees of freedom = `n-p`, where `p` includes the intercept).  
Reject H₀ if:  
- `|t_j| > t_{n-p;1-α/2]`, or  
- p-value `< α`. $t= \pm 1.96$ corresponds to $p =0.05$

## Interview tip  
> “We look at each coefficient’s t‑statistic to see if that feature significantly adds explanatory power. In a quant setting, this helps us decide which factors to include in our predictive model.”

---

## F‑Test for Overall Model Fit

## Purpose  
Test whether at least one predictor in your model has explanatory power (i.e., whether the regression as a whole is meaningful).

## Null hypothesis  
- **H₀:** `β_1 = β_2 = … = β_{p-1} = 0` (no linear relationship whatsoever)  
- **H₁:** at least one `β_j ≠ 0`

## Test statistic  
```
 F = (SSR/(p-1)) / (SSE/(n-p))
    = (R^2/(p-1)) / ((1-R^2)/(n-p))
```  
- SSR = regression sum of squares.  
- SSE = error sum of squares.  
- `R^2` = coefficient of determination.  
- `p` = number of parameters (including intercept).

## Distribution and decision rule  
Under H₀,  
```
 F ~ F_{p-1, n-p}
```  
Reject H₀ if:  
- `F > F_{p-1, n-p;1-α}`, or  
- p-value `< α`.

## Interview tip  
> “The F‑test tells us if our model collectively explains variance in the dependent variable. In a quant research context, this is a quick check on whether any of our candidate factors—taken together—have predictive value before we dive deeper into individual t‑tests or out‑of-sample backtests.”

---

## Putting It Together

```
“In linear regression, I use a t‑test on each coefficient beta_hat_j to see if that single factor has a statistically significant relationship with the target—rejecting H₀: beta_j=0 if the t‑statistic’s p‑value is below my chosen alpha. Then I perform an F‑test on the whole model to ensure that, as a group, the regressors explain a non‑zero fraction of variance—rejecting H₀: beta_1=…=beta_{p-1}=0 if the F‑statistic is large enough. Together, these tests guide factor selection and give confidence that the model is not just fitting noise.”
```


## Homoscedasticity and heteroscedasticity

**Homoscedasticity** assumes constant error variance, which underpins the classic OLS standard‐error formulas. 
$$Var(\epsilon_i | X_i) = \sigma^2  \forall i\in \{1,...,n\}.$$
**Heteroscedasticity** means the error variance changes with predictors—OLS estimates remain unbiased but inference (t‑tests and confidence intervals) is invalid because the usual variance formulas are wrong.
$$Var(\epsilon_i | X_i) = \sigma_i^2,  \sigma_i^2 \not= constant.$$

When heteroscedasticity is present, I can switch to weighted least squares, weighting observations by the inverse of their estimated variance to restore efficiency and correct inference..


## Random Forest feature importances

Random Forest feature importances quantify how much each predictor contributes to reducing prediction error across all the trees in the forest. For regression trees (as in our example), the most common measure is the Mean Decrease in Impurity (MDI):

- At each split in each decision tree, the algorithm chooses the feature and split‐point that most reduces the node’s mean‐squared error (MSE).
  ```
  importance_data[node.feature] += (
    node.weighted_n_node_samples * node.impurity —       
    left.weighted_n_node_samples * left.impurity — 
    right.weighted_n_node_samples * right.impurity)
  ```

- For each feature, you sum up all those reductions in MSE across every split in every tree where that feature was used.

- Normalize these sums so they add up to 1. The result is the vector $[I_1,..,I_p]$ where $I_j$ is the importance of feature $j$.

- **Example**
| Feature  | Importance (\%) |
|:---------|:----------------|
| lag_1    |            78.9 |
| lag_2    |            9.5  |
| lag_3    |            3.0  |
| lag_7    |            3.8  |
| time     |            4.8  |
Interpretation: ``lag_1`` (the previous day’s value) explains nearly 80 % of the model’s total impurity reduction, so it’s by far the most predictive. The calendar feature time comes in last.

- Why it matters

  - Model insight: You immediately see which lags or external features drive your forecasts.

  - Feature selection: You could drop very low‐importance variables to simplify the model.

  - Diagnostics: If you expect ``seasonal_dummy`` or a macro variable to matter but see near-zero importance, it signals a mismatch.
 
- In practice, you can also compute permutation importances—which measure how much shuffling values of each feature degrades out-of-sample accuracy—to get a more robust picture, especially when features are correlated.

