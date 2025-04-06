# Machine-Learning

## Table of Contents
- [Introduction](#introduction)
- [Data Analysis Techniques](#data-analysis-techniques)
- [Python Code Examples](#python-code-examples)


### How to deal with missing values
You're aiming to keep the data as representative and unbiased as possible. The method chosen should not change the data distribution too much.
+ Method : ffill, bfill, fill by mean, fill by median, dropna
+ % Missing	What You Might Do
< 5%	Probably safe to fill or drop rows
5–15%	Imputation usually fine, especially if column is important
15–30%	Think carefully. Maybe keep, maybe drop. Use your domain knowledge.
>30%	Often better to drop the column (if not essential), or replace it with a model prediction
>50%	Usually not worth saving unless the feature is very important and imputation is smart
