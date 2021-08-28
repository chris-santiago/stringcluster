# string-cluster

## Install

Create a virtual environment with Python 3.9 and install from git:

```bash
pip install git+https://github.com/chris-santiago/stringcluster.git
```

## Use

### Preliminaries

This example shows how to use `StringCluster` to deduplicate a list of public company names.  The example dataset is a series of company names and their respective variations.  

`StringCluster` uses Tf-Idf vectorization to tokenize each element in a series of strings and normalize the count of each n-gram token. It then uses this transformation to construct a cosine similarity matrix by computing the linear kernel for the vector representations of each data observation. `StringCluster` can compare cosine similarity to either itself or a master list of strings to de-duplicate the original series.


```python
import re

import pandas as pd

from stringcluster import StringCluster
```

### Data

As mentioned, the example dataset is a series of company names (strings). To illustrate, we'll pull out all samples that contain the string "FACEBOOK"; we have 11 unique versions for this single company.


```python
data = pd.read_csv('../data/companies.csv')
data.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>company</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MICROSOFT CORP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>APPLE INC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FACEBOOK INC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ISHARES TR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORACLE CORP</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ALPHABET INC - A</td>
    </tr>
    <tr>
      <th>6</th>
      <td>JOHNSON &amp; JOHNSON</td>
    </tr>
    <tr>
      <th>7</th>
      <td>WESTERN DIGITAL CORP</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AMAZON.COM INC</td>
    </tr>
    <tr>
      <th>9</th>
      <td>VISA INC</td>
    </tr>
  </tbody>
</table>
</div>




```python
companies = data['company']
mask = data['company'].str.contains('FACEBOOK')
facebook = data['company'][mask]
print(f'Number of unique version: {facebook.nunique()}')
facebook
```

    Number of unique version: 11





    2                                           FACEBOOK INC
    408     FACEBOOK INC            CLASS                  A
    474                                    FACEBOOK INC CL A
    998                                           FACEBOOK-A
    1042                                FACEBOOK INC CLASS A
    1101                                      FACEBOOK INC A
    1448                                      FACEBOOK INC-A
    3020                                FACEBOOK INC COM NPV
    3626                                     FACEBOOK INC -A
    3638                                            FACEBOOK
    4340                                      FACEBOOK, INC.
    Name: company, dtype: object



### Deduplicating

As mentioned, `StringCluster` can be used with or without a "master" list of string representations, depending on the use case. A master list is provided as the `y` parameter in the `.fit_transform()` method. This can be useful if user have a designated set of representations that they wish to group each sample under.

#### Without a master list

Let's first take a look at use **without** a master list.  The `StringCluster` transformer takes three parameters:

|Parameter|Type|Description|
|---------|----|-----------|
|`ngram_size`|int|Size of ngrams to be extracted; default 2.|
|`threshold`|float|Threshold to determine similarities; must be between [0, 1]; default 0.8.|
|`stop_tokens`|str|RegEx pattern to remove during tokenization; default `r'[\W_]+'`|

Although we're using Tf-Idf vectorization, and common tokens will have less effect, it's still important to provide a list of domain-specific stop tokens. In this case, we'll remove special characters, white space and any word that relates to "corporation", "incorporated", etc., prior to Tf-Idf vectorization-- these variations within a company's name are meaningless.

After fitting the `StringCluster` object and transforming the data, we see that all 11 variations of "Facebook" have consolidated to "FACEBOOK INC". 

**Of note: When using `StringCluster` without a master list, the transformer will default to replacing variations of a string representation with the first variation seen-- in the case, "FACEBOOK INC".**


```python
STOP_TOKENS = r'[\W_]+|(corporation$)|(corp.$)|(corp$)|(incorporated$)|(inc.$)|(inc$)|(company$)|(common$)|(com$)'

cluster = StringCluster(ngram_size=2, threshold=0.7, stop_tokens=STOP_TOKENS)
labels = cluster.fit_transform(data['company'])
```


```python
labels[facebook.index]
```




    2       FACEBOOK INC
    408     FACEBOOK INC
    474     FACEBOOK INC
    998     FACEBOOK INC
    1042    FACEBOOK INC
    1101    FACEBOOK INC
    1448    FACEBOOK INC
    3020    FACEBOOK INC
    3626    FACEBOOK INC
    3638    FACEBOOK INC
    4340    FACEBOOK INC
    Name: company, dtype: object



#### With a master list

Let's take a look at use with a master list.  As mentioned, the mast list is passed as the `y` parameter in the `.fit()` and `fit_transform()` methods.  In this case, each string in the series is compared against the master list and replaced with the representation in the master list with which it exhibits the highest cosine similarity.


```python
TEST_SERIES = pd.Series(
        ['Johnson & Johnson, Inc.', 'Johnson & Johnson Inc.', 'Johnson & Johnson Inc',
         'Johnson & Johnson', 'Intel Corp', 'Intel Corp.', 'Intel Corporation', 'Google',
         'Apple', 'Amazon', 'Amazon Inc', 'Comcast Inc.', 'Comcast Corp']
    )
MASTER = ['Johnson & Johnson', 'Intel Corp', 'Google', 'Apple Inc', 'Amazon', 'Comcast']

STOP_TOKENS = r'[\W_]+|(corporation$)|(corp.$)|(corp$)|(incorporated$)|(inc.$)|(inc$)|(company$)|(common$)|(com$)'

cluster = StringCluster(ngram_size=2, stop_tokens=STOP_TOKENS)
labels = cluster.fit_transform(TEST_SERIES, MASTER)
```


```python
labels
```




    0     Johnson & Johnson
    1     Johnson & Johnson
    2     Johnson & Johnson
    3     Johnson & Johnson
    4            Intel Corp
    5            Intel Corp
    6            Intel Corp
    7                Google
    8             Apple Inc
    9                Amazon
    10               Amazon
    11              Comcast
    12              Comcast
    dtype: object



### Trialing Different Threshold Values

The `StringCluster` transformer is sensitive to the `threshold` parameter (especially without a master list), as this controls how matches are flagged, based on their cosine similarity.  Let's take a look at how varying levels of the `threshold` parameter affect results on our Facebook example.


```python
thresh = 0.7
while thresh < 1:
    cluster = StringCluster(ngram_size=2, threshold=thresh, stop_tokens=STOP_TOKENS)
    labels = cluster.fit_transform(data['company'])
    print(f'Threshold: {thresh}')
    print('----------------------------------------')
    print(labels[facebook.index])
    print('========================================')
    thresh += 0.05
```

    Threshold: 0.7
    ----------------------------------------
    2       FACEBOOK INC
    408     FACEBOOK INC
    474     FACEBOOK INC
    998     FACEBOOK INC
    1042    FACEBOOK INC
    1101    FACEBOOK INC
    1448    FACEBOOK INC
    3020    FACEBOOK INC
    3626    FACEBOOK INC
    3638    FACEBOOK INC
    4340    FACEBOOK INC
    Name: company, dtype: object
    ========================================
    Threshold: 0.75
    ----------------------------------------
    2               FACEBOOK INC
    408             FACEBOOK INC
    474             FACEBOOK INC
    998             FACEBOOK INC
    1042            FACEBOOK INC
    1101            FACEBOOK INC
    1448            FACEBOOK INC
    3020    FACEBOOK INC COM NPV
    3626            FACEBOOK INC
    3638            FACEBOOK INC
    4340            FACEBOOK INC
    Name: company, dtype: object
    ========================================
    Threshold: 0.8
    ----------------------------------------
    2                                           FACEBOOK INC
    408     FACEBOOK INC            CLASS                  A
    474                                         FACEBOOK INC
    998                                         FACEBOOK INC
    1042    FACEBOOK INC            CLASS                  A
    1101                                        FACEBOOK INC
    1448                                        FACEBOOK INC
    3020                                FACEBOOK INC COM NPV
    3626                                        FACEBOOK INC
    3638                                        FACEBOOK INC
    4340                                        FACEBOOK INC
    Name: company, dtype: object
    ========================================
    Threshold: 0.8500000000000001
    ----------------------------------------
    2                                           FACEBOOK INC
    408     FACEBOOK INC            CLASS                  A
    474     FACEBOOK INC            CLASS                  A
    998                                         FACEBOOK INC
    1042    FACEBOOK INC            CLASS                  A
    1101                                        FACEBOOK INC
    1448                                        FACEBOOK INC
    3020                                FACEBOOK INC COM NPV
    3626                                        FACEBOOK INC
    3638                                        FACEBOOK INC
    4340                                        FACEBOOK INC
    Name: company, dtype: object
    ========================================
    Threshold: 0.9000000000000001
    ----------------------------------------
    2                                           FACEBOOK INC
    408     FACEBOOK INC            CLASS                  A
    474     FACEBOOK INC            CLASS                  A
    998                                         FACEBOOK INC
    1042    FACEBOOK INC            CLASS                  A
    1101                                   FACEBOOK INC CL A
    1448                                   FACEBOOK INC CL A
    3020                                FACEBOOK INC COM NPV
    3626                                   FACEBOOK INC CL A
    3638                                        FACEBOOK INC
    4340                                        FACEBOOK INC
    Name: company, dtype: object
    ========================================
    Threshold: 0.9500000000000002
    ----------------------------------------
    2                                           FACEBOOK INC
    408     FACEBOOK INC            CLASS                  A
    474                                    FACEBOOK INC CL A
    998                                         FACEBOOK INC
    1042    FACEBOOK INC            CLASS                  A
    1101                                      FACEBOOK INC A
    1448                                      FACEBOOK INC A
    3020                                FACEBOOK INC COM NPV
    3626                                      FACEBOOK INC A
    3638                                        FACEBOOK INC
    4340                                        FACEBOOK INC
    Name: company, dtype: object
    ========================================

