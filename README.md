# Machine-Learning

## Table of Contents
- [How to deal with missing values](#How to deal with missing values)
- [Data Analysis Techniques](#data-analysis-techniques)
- [Python Code Examples](#python-code-examples)


### How to deal with missing values
You're aiming to keep the data as representative and unbiased as possible. The method chosen should not change the data distribution too much.
+ Methods : ffill, bfill, fill by mean, fill by median, dropna
+ Time series : prioritize ffill. 
+ < 5% Missing : probably safe to fill or drop rows. >30%	Often better to drop the column (if not essential).

```python
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")```

i finish
