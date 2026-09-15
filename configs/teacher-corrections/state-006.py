import torch
import torch.nn.functional as F


class Attention:
    def __init__(self, config):
        self.heads = config["heads"]

    def topology(self):
        return "attention"

    def load_parameters(self, parameters):
        self.qkv = torch.from_numpy(parameters["qkv"].copy()).T.contiguous()
        self.out = torch.from_numpy(parameters["out"].copy()).T.contiguous()

    def export_parameters(self):
        return {"qkv": self.qkv, "out": self.out}

    def run(self, x, draws):
        batch, sequence, width = x.shape
        head_dim = width // self.heads
        q, k, v = [part.reshape(batch, sequence, self.heads, head_dim).transpose(1, 2)
                   for part in F.linear(x, self.qkv).chunk(3, dim=-1)]
        scores = q @ k.transpose(-2, -1) / head_dim ** 0.5
        mask = torch.ones(sequence, sequence, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~mask, -torch.inf).softmax(dim=-1)
        layer = (weights @ v).transpose(1, 2).contiguous().reshape(batch, sequence, width)
        return {"layer": layer, "output": F.linear(layer, self.out)}


def build(config):
    return Attention(config)
