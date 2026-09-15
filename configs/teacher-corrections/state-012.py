import torch


def build(config):
    class LayerNorm:
        def __init__(self, config):
            self.config = config
            self.scale = None
            self.bias = None
            self.weight = None

        def topology(self):
            return "layernorm"

        def load_parameters(self, parameters):
            self.scale = torch.tensor(parameters["scale"], dtype=torch.float32)
            self.bias = torch.tensor(parameters["bias"], dtype=torch.float32)
            self.weight = torch.tensor(parameters["weight"], dtype=torch.float32)

        def export_parameters(self):
            return {"scale": self.scale, "bias": self.bias, "weight": self.weight.T}

        def run(self, x, draws):
            x = x.to(dtype=torch.float32)
            centered = x - x.mean(dim=-1, keepdim=True)
            variance = centered.pow(2).mean(dim=-1, keepdim=True)
            normalized = centered / torch.sqrt(variance + self.config["epsilon"])
            layer = normalized * self.scale + self.bias
            output = x + layer @ self.weight
            return {"layer": layer, "output": output}

    return LayerNorm(config)
