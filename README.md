# jaxmetalog

JAX Implementation of the Metalog Distribution. Also
implements the LogMetalog (lower bounded version),
NegativeLogMetalog (upper bounded) and LogitMetalog (both
bounds). See the [Wikipedia](https://en.wikipedia.org/wiki/Metalog_distribution)
article for details on each of these.

## Differences from other code

Parameters are estimated via Gradient Descent on the
squared error between the distribution and empirical CDF.
This is why I employed JAX. In other words, this code does
not estimate the exact solution.

## Choosing the number of terms

By default, the Bayesian Information Criterion is optimized
choosing the number of terms from within the [5, 20] range.

## How to use?

Check the notebooks folder.
