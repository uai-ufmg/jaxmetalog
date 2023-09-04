# -*- coding: utf8 -*-


import jax
import jax.numpy as jnp


@jax.jit
def M_k(y, weights):
    l = jnp.log(y / (1 - y))  # noqa: E741
    d = (y - 0.5)

    rv = weights[0] + weights[1] * l + weights[2] * d * l + weights[3] * d  # noqa: E501

    def loop_body(i, rv):
        k = i + 1
        return jax.lax.cond(
            k % 2 != 0,
            lambda: rv + weights[i] * jnp.power(d, (k - 1) / 2),  # noqa: E501
            lambda: rv + weights[i] * jnp.power(d, -1 + k / 2) * l  # noqa: E501
        )

    return jax.lax.fori_loop(
        4, weights.shape[0], loop_body, rv
    )


@jax.jit
def m_k(y, weights):
    l = jnp.log(y / (1 - y))  # noqa: E741
    d = y - 0.5
    p = y * (1 - y)

    rv = jnp.power(
        (weights[1] / p) + (weights[2] * ((d / p) + l)) + weights[3], -1.0  # noqa: E501
    )

    def loop_body(i, rv):
        k = i + 1
        rv = jax.lax.cond(
            k % 2 != 0,
            lambda: (1.0 / rv) + weights[i] * 0.5 * (k - 1) * jnp.power(d, (k - 3) / 2),  # noqa: E501
            lambda: (1.0 / rv) + weights[i] * ((jnp.power(d, -1 + k / 2) / p) + (-1 + k / 2) * jnp.power(d, -2 + k / 2) * l)  # noqa: E501,E741
        )
        return jnp.power(rv, -1.0)

    return jax.lax.fori_loop(
        4, weights.shape[0], loop_body, rv
    )


@jax.jit
def mse(weights, x, y):
    x_hat = M_k(y, weights)
    return jnp.power(x - x_hat, 2).mean()


dmse = jax.grad(mse)


@jax.jit
def gd(w_init, x, y, lr, n_iter):
    return jax.lax.fori_loop(
        0, n_iter,
        lambda _, w: w - lr * dmse(w, x, y),
        w_init
    )


def bic(weights, y, m):
    k = weights.shape[0]
    n = y.shape[0]
    return k * jnp.log(n) + \
        -2 * jnp.log(m(y, weights)).sum()


def fit(x, y, m, lr=0.1, n_iter=200, k_min=5, k_max=20):
    best = jnp.inf
    rv = None
    for k in range(k_min, k_max):
        w_init = jnp.ones(k)
        w = gd(w_init, x, y, lr, n_iter)
        score = bic(w, y, m)
        if score < best:
            rv = w
            best = score
    return rv
