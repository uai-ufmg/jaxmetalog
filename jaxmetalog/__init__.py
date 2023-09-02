# -*- coding: utf8 -*-


from .base import fit as _fit
from .base import M_k as _M_k
from .base import m_k as _m_k

from jax import Array

from numpy.typing import ArrayLike as NPArrayLike

from statsmodels.distributions import ECDF

from typing import Callable
from typing import Tuple


import jax
import jax.numpy as jnp

import numpy as np


KeyArray = Array | jax.random.PRNGKeyArray


class _BaseMetalog(object):

    def __init__(
        self,
        z: Callable[[NPArrayLike], NPArrayLike],
        M: Callable[[Array, Array], Array],
        m: Callable[[Array, Array], Array],
        learning_rate: float = 0.1,
        number_of_iterations: int = 200
    ) -> None:
        self._weights = None
        self.z = z
        self.M = M
        self.m = m
        self._lr = learning_rate
        self._n_iter = number_of_iterations

    def fit(
        self,
        data: Array | NPArrayLike
    ) -> None:

        # ECDF is numpy code.
        data = self.z(np.asanyarray(data))
        ecdf = ECDF(data)

        # From here, everything is JAX
        x = jnp.array(ecdf.x[1:-1])
        y = jnp.array(ecdf.y[1:-1])
        self._weights = _fit(x, y, self._lr, self._n_iter)

        if self._weights is None:
            raise AttributeError(
                'Failed! Try lowering the learning rate.'
            )

    def sample(
        self,
        key: KeyArray,
        n_samples: int
    ) -> Array:
        if self._weights is None:
            raise UnboundLocalError('Call fit first')

        y = jax.random.uniform(key, shape=(n_samples, ))
        return self.M(y, self._weights)

    def pdf(
        self,
        data: Array | NPArrayLike
    ) -> Array:
        if self._weights is None:
            raise UnboundLocalError('Call fit first')

        data = self.z(np.asanyarray(data))
        ecdf = ECDF(data)
        y = jnp.array(ecdf.y[1:-1])
        return self.m(y, self._weights)

    def ppf(
        self,
        data: Array | NPArrayLike
    ) -> Tuple[Array, Array]:
        if self._weights is None:
            raise UnboundLocalError('Call fit first')

        data = self.z(np.asanyarray(data))
        ecdf = ECDF(data)
        y = jnp.array(ecdf.y[1:-1])
        return self.M(y, self._weights), y


class Metalog(_BaseMetalog):

    def __init__(
        self,
        learning_rate: float = 0.1,
        number_of_iterations: int = 200
    ) -> None:
        def M(y: Array, weights: Array) -> Array:
            return _M_k(y, weights)

        def m(y: Array, weights: Array) -> Array:
            return _m_k(y, weights)

        super().__init__(
            lambda x: x, M, m,
            learning_rate, number_of_iterations
        )


class LogMetalog(_BaseMetalog):

    def __init__(
        self,
        b_lower: float,
        learning_rate: float = 0.1,
        number_of_iterations: int = 200
    ) -> None:
        self._b_lower = b_lower

        def M(y: Array, weights: Array) -> Array:
            e = jnp.exp(_M_k(y, weights))
            return self._b_lower + e

        def m(y: Array, weights: Array) -> Array:
            e = jnp.exp(-1.0 * _M_k(y, weights))
            return _m_k(y, weights) * e

        def z(data: NPArrayLike) -> NPArrayLike:
            b_l = np.asanyarray(self._b_lower)
            return np.log(data - b_l)

        super().__init__(
            z, M, m, learning_rate, number_of_iterations
        )


class NegativeLogMetalog(_BaseMetalog):

    def __init__(
        self,
        b_upper: float,
        learning_rate: float = 0.1,
        number_of_iterations: int = 200
    ) -> None:
        self._b_upper = b_upper

        def M(y: Array, weights: Array) -> Array:
            e = jnp.exp(-1.0 * _M_k(y, weights))
            return self._b_upper - e

        def m(y: Array, weights: Array) -> Array:
            e = jnp.exp(_M_k(y, weights))
            return _m_k(y, weights) * e

        def z(data: NPArrayLike) -> NPArrayLike:
            b_u = np.asanyarray(self._b_upper)
            return -np.log(b_u - data)

        super().__init__(
            z, M, m, learning_rate, number_of_iterations
        )


class LogitMetalog(_BaseMetalog):

    def __init__(
        self,
        b_lower: float,
        b_upper: float,
        learning_rate: float = 0.1,
        number_of_iterations: int = 200
    ) -> None:
        self._b_lower = b_lower
        self._b_upper = b_upper

        def M(y: Array, weights: Array) -> Array:
            e = jnp.exp(_M_k(y, weights))
            num = self._b_lower + self._b_upper * e
            den = 1.0 + e
            return num / den

        def m(y: Array, weights: Array) -> Array:
            e = jnp.exp(_M_k(y, weights))

            num = (1.0 + e) ** 2
            den = (self._b_upper - self._b_lower) * e

            return _m_k(y, weights) * num / den

        def z(data: NPArrayLike) -> NPArrayLike:
            b_l = np.asanyarray(self._b_lower)
            b_u = np.asanyarray(self._b_upper)
            return np.log((data - b_l) / (b_u - data))

        super().__init__(
            z, M, m, learning_rate, number_of_iterations
        )
