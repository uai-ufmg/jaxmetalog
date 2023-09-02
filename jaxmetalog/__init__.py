# -*- coding: utf8 -*-


import jax
import jax.numpy as jnp


from .base import fit as _fit
from .base import M_k
from .base import m_k


class Metalog(object):

    def fit(self, data):
        ecdf = ECDF(data)
        x = jnp.array(ecdf.x[1:-1])
        y = jnp.array(ecdf.y[1:-1])
        self._weights = _fit(x, y)

    def sample(self, key, n_samples):
        y = jax.random.uniform(key, shape=(n_samples, ))
        return M_k(y, self._weights)

    def pdf(self, data):
        ecdf = ECDF(data)
        y = jnp.array(ecdf.y[1:-1])
        return m_k(y, self._weights)

    def ppf(self, data):
        ecdf = ECDF(data)
        y = jnp.array(ecdf.y[1:-1])
        return M_k(y, self._weights)