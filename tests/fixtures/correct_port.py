"""Evaluator calibration oracle. Never copy into policy workspaces."""
import torch
import torch.nn.functional as F


class Port:
    def __init__(self, config):
        self.c = config

    def topology(self):
        return self.c["template"]

    def load_parameters(self, parameters):
        transposed = {"w1", "w2", "weight", "qkv", "out"}
        self.p = {k: torch.from_numpy(v.copy()).permute(3, 2, 0, 1).contiguous() if k == "kernel"
                  else torch.from_numpy(v.copy()).T.contiguous() if k in transposed
                  else torch.from_numpy(v.copy()) for k, v in parameters.items()}

    def export_parameters(self):
        return self.p

    def run(self, x, draws):
        p, c, name = self.p, self.c, self.topology()
        if name == "mlp":
            layer = F.relu(F.linear(x, p["w1"], p["b1"]))
            output = F.linear(layer, p["w2"], p["b2"])
        elif name == "cnn":
            layer = F.conv2d(x.permute(0, 3, 1, 2), p["kernel"], p["bias"], padding=1).permute(0, 2, 3, 1)
            output = layer.relu().mean((1, 2))
        elif name == "layernorm":
            layer = F.layer_norm(x, (c["width"],), p["scale"], p["bias"], c["epsilon"])
            output = x + F.linear(layer, p["weight"])
        elif name in ("attention", "rope_cache"):
            q, k, v = self.qkv(x, c["offset"])
            scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
            mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).tril()
            attended = (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v).transpose(1, 2).reshape(x.shape)
            layer = torch.stack((k, v)) if name == "rope_cache" else attended
            output = F.linear(attended, p["out"])
        else:
            layer = F.linear(x, p["weight"], p["bias"])
            output = (draws[..., None] > layer.softmax(-1).cumsum(-1)).sum(-1).clamp_max(c["hidden"] - 1)
        return {"layer": layer, "output": output}

    def qkv(self, x, offset):
        b, n, w = x.shape
        heads = self.c["heads"]
        d = w // heads
        q, k, v = [z.reshape(b, n, heads, d).transpose(1, 2)
                   for z in F.linear(x, self.p["qkv"]).chunk(3, -1)]
        if self.topology() == "rope_cache":
            angles = (torch.arange(n) + offset)[:, None] * 10000.0 ** (-torch.arange(0, d, 2) / d)

            def rotate(z):
                even, odd = z[..., ::2], z[..., 1::2]
                return torch.stack((even * angles.cos() - odd * angles.sin(),
                                    even * angles.sin() + odd * angles.cos()), -1).reshape(z.shape)

            q, k = rotate(q), rotate(k)
        return q, k, v

    def run_cached(self, x, draws):
        keys, values, outputs = [], [], []
        for i in range(x.shape[1]):
            q, k, v = self.qkv(x[:, i:i + 1], self.c["offset"] + i)
            keys.append(k)
            values.append(v)
            scores = q @ torch.cat(keys, 2).transpose(-1, -2) / q.shape[-1] ** 0.5
            attended = (scores.softmax(-1) @ torch.cat(values, 2)).transpose(1, 2).reshape(x.shape[0], 1, x.shape[-1])
            outputs.append(F.linear(attended, self.p["out"]))
        return torch.cat(outputs, 1)


def build(config):
    return Port(config)
