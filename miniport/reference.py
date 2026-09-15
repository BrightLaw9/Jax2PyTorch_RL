"""Trusted JAX implementations; all numerical fixtures use float32."""
import jax
import jax.numpy as jnp
import numpy as np


def fixture(task, seed):
    key = jax.random.PRNGKey(seed)

    def normal(shape):
        nonlocal key
        key, subkey = jax.random.split(key)
        return jax.random.normal(subkey, shape, dtype=jnp.float32) * 0.2

    b, n, w, h = task.batch, task.length, task.width, task.hidden
    x = normal((b, n, w))
    p = {}
    if task.template == "mlp":
        p = dict(w1=normal((w, h)), b1=normal((h,)),
                 w2=normal((h, w)), b2=normal((w,)))
    elif task.template == "cnn":
        x = normal((b, n, n + 1, w))
        p = dict(kernel=normal((3, 3, w, h)), bias=normal((h,)))
    elif task.template == "layernorm":
        p = dict(scale=normal((w,)) + 1, bias=normal((w,)),
                 weight=normal((w, w)))
    elif task.template in ("attention", "rope_cache"):
        p = dict(qkv=normal((w, 3 * w)), out=normal((w, w)))
    elif task.template == "sampling":
        p = dict(weight=normal((w, h)), bias=normal((h,)))
    else:
        raise ValueError(task.template)
    # The split sequence is part of the sampling contract. The candidate receives
    # these draws, not a claim that torch.manual_seed reproduces JAX's RNG.
    key, subkey = jax.random.split(key)
    draws = jax.random.uniform(subkey, (b, n), minval=1e-5, maxval=1 - 1e-5)
    return {k: np.asarray(v).copy() for k, v in p.items()}, np.asarray(x).copy(), np.asarray(draws).copy()


def mapped_parameters(task, p):
    transpose = {"mlp": {"w1", "w2"}, "layernorm": {"weight"},
                 "attention": {"qkv", "out"}, "rope_cache": {"qkv", "out"},
                 "sampling": {"weight"}}.get(task.template, set())
    return {k: (v.transpose(3, 2, 0, 1) if k == "kernel" else
                v.T if k in transpose else v).copy() for k, v in p.items()}


def reference(task, parameters, inputs, draws):
    p = {k: jnp.asarray(v) for k, v in parameters.items()}
    x = jnp.asarray(inputs)
    name = task.template
    if name == "mlp":
        layer = jnp.maximum(x @ p["w1"] + p["b1"], 0)
        output = layer @ p["w2"] + p["b2"]
    elif name == "cnn":
        layer = jax.lax.conv_general_dilated(
            x, p["kernel"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + p["bias"]
        output = jnp.maximum(layer, 0).mean(axis=(1, 2))
    elif name == "layernorm":
        centered = x - x.mean(axis=-1, keepdims=True)
        layer = centered / jnp.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + task.epsilon)
        layer = layer * p["scale"] + p["bias"]
        output = x + layer @ p["weight"]
    elif name in ("attention", "rope_cache"):
        b, n, w = x.shape
        d = w // task.heads
        q, k, v = [z.reshape(b, n, task.heads, d).transpose(0, 2, 1, 3)
                   for z in jnp.split(x @ p["qkv"], 3, axis=-1)]
        if name == "rope_cache":
            positions = jnp.arange(n) + task.offset
            angles = positions[:, None] * 10000.0 ** (-jnp.arange(0, d, 2) / d)

            def rotate(z):
                even, odd = z[..., ::2], z[..., 1::2]
                return jnp.stack((even * jnp.cos(angles) - odd * jnp.sin(angles),
                                  even * jnp.sin(angles) + odd * jnp.cos(angles)), axis=-1).reshape(z.shape)

            q, k = rotate(q), rotate(k)
            # Expose the ordered KV cache as the layer output. Candidate must also
            # return the incremental output, tested against full causal attention.
            layer = jnp.stack((k, v), axis=0)
        scores = (q @ k.swapaxes(-2, -1)) / jnp.sqrt(float(d))
        scores = jnp.where(jnp.tril(jnp.ones((n, n), dtype=bool)), scores, -jnp.inf)
        weights = jax.nn.softmax(scores, axis=-1)
        attended = (weights @ v).transpose(0, 2, 1, 3).reshape(b, n, w)
        if name == "attention":
            layer = attended
        output = attended @ p["out"]
    else:
        layer = x @ p["weight"] + p["bias"]
        cdf = jnp.cumsum(jax.nn.softmax(layer, axis=-1), axis=-1)
        output = jnp.minimum((jnp.asarray(draws)[..., None] > cdf).sum(axis=-1), task.hidden - 1)
    return {"layer": np.asarray(layer), "output": np.asarray(output)}
