"""Agent-visible task requirements, without executable JAX references."""
EQUATIONS = {
    "mlp": "layer = ReLU(x @ w1 + b1); output = layer @ w2 + b2.",
    "cnn": "Input is NHWC. Kernel is HWIO, 3x3, stride 1, SAME padding. layer = conv(x, kernel) + bias in NHWC; output = mean(ReLU(layer), spatial axes).",
    "layernorm": "Normalize the last axis using population variance and config epsilon, then multiply by scale and add bias. This is layer. output = x + layer @ weight.",
    "attention": "Project x @ qkv, split contiguous Q, K, V on last axis; reshape each to [batch, heads, sequence, width/heads]. Apply causal attention softmax(Q K^T / sqrt(head_dim)), including diagonal. layer is attended values merged back to [batch, sequence, width]; output = layer @ out.",
    "rope_cache": "Use packed QKV and causal attention as in attention. Rotate adjacent even/odd Q and K coordinates by angle (position + offset) * 10000^(-2i/head_dim). layer = stack(K_rotated, V) with shape [2,batch,heads,sequence,head_dim]. output is merged causal attention @ out. run_cached must process one token at a time, appending K/V in sequence order and advancing offsets; return the full concatenated output. Each call starts an empty cache.",
    "sampling": "layer = x @ weight + bias. output = count(draw > cumulative softmax(layer), last axis), clipped to hidden-1. Draws are provided from a JAX PRNG split stream. Return integer indices; do not substitute a native PyTorch RNG seed.",
}


TENSOR_CONTRACTS = {
    "mlp": "layer: [batch, sequence, hidden], AFTER ReLU. output: [batch, sequence, width], final affine result without activation.",
    "cnn": "layer: [batch, height, width, output_channels] (NHWC), the exact raw convolution + bias result BEFORE ReLU. Returning ReLU(layer) as layer is incorrect even if output is otherwise correct. output: [batch, output_channels], spatial mean of ReLU(raw layer). Keep the raw layer for the returned dictionary; compute the activated tensor separately for output. A PyTorch conv2d result is NCHW [batch, output_channels, height, width], so its spatial mean is mean(dim=(2,3)); mean(dim=(1,2)) is wrong because dimension 1 is the output-channel axis and would leave a spatial-width result. If you first convert the activated tensor to NHWC, its spatial mean is mean(dim=(1,2)). Export kernel: [output_channels, input_channels, kernel_height, kernel_width] (OIHW). Input and loaded kernel remain NHWC and HWIO respectively.",
    "layernorm": "layer: [batch, sequence, width], normalized input after scale and bias. output: same shape, x + layer @ weight; preserve the residual.",
    "attention": "layer: [batch, sequence, width], attended values AFTER merging heads, BEFORE the output projection. output: same shape, layer @ out.",
    "rope_cache": "layer: [2, batch, heads, sequence, head_dim], stacked rotated K and unrotated V. output and run_cached: [batch, sequence, width], after output projection.",
    "sampling": "layer: [batch, sequence, hidden], raw logits BEFORE softmax. output: [batch, sequence], integer sampled indices using the supplied draws.",
}


PARAMETER_CONTRACTS = {
    "mlp": "Loaded w1:[width,hidden], b1:[hidden], w2:[hidden,width], b2:[width]. Export w1/w2 transposed; biases unchanged.",
    "cnn": "input_channels=config['width']; output_channels=config['hidden']. There are NO config keys named input_channels or output_channels. Loaded kernel:[3,3,width,hidden] (HWIO), bias:[hidden]. Export kernel:[hidden,width,3,3] (OIHW), bias unchanged. Keep loaded kernel dimensions; reshape does not reorder axes. Spatial sizes come from x.shape; never hard-code the example batch or image size. PyTorch conv2d consumes NCHW and OIHW; spatial axes are (2,3) in NCHW versus (1,2) in NHWC.",
    "layernorm": "Loaded scale:[width], bias:[width], weight:[width,width]. Export only weight transposed. Normalize with population variance (correction=0), using config['epsilon'].",
    "attention": "heads=config['heads']; head_dim=config['width']//heads. Loaded qkv:[width,3*width], out:[width,width]; export both transposed. Q/K/V are contiguous chunks of the projection's last axis. Required shape ledger: projected QKV [batch,sequence,3*width]; each contiguous split [batch,sequence,width]; reshape each split to [batch,sequence,heads,head_dim]; permute each to [batch,heads,sequence,head_dim]; attention scores [batch,heads,sequence,sequence]; causal mask [sequence,sequence] or [1,1,sequence,sequence], broadcast over batch and heads; attended values [batch,heads,sequence,head_dim]; permute to [batch,sequence,heads,head_dim] and reshape to merged layer [batch,sequence,width]. Do not multiply attention while Q/K/V remain [batch,sequence,heads,head_dim]. Softmax uses the final key-sequence axis; future positions are masked. No dropout.",
    "rope_cache": "heads=config['heads']; head_dim=config['width']//heads; position starts at config['offset']. Loaded qkv:[width,3*width], out:[width,width]; export both transposed. Cache length advances positions per token, and resets between calls.",
    "sampling": "Loaded weight:[width,hidden], bias:[hidden]. Export weight transposed, bias unchanged. hidden is the number of categories. Draws have shape [batch,sequence]; output indices must be int32 or int64.",
}


SHAPE_LEDGERS = {
    "attention": [
        "qkv projection: [batch, sequence, 3*width]",
        "contiguous q/k/v splits: [batch, sequence, width]",
        "reshape each split: [batch, sequence, heads, head_dim]",
        "permute q/k/v before matmul: [batch, heads, sequence, head_dim]",
        "q @ k.transpose(-2,-1): [batch, heads, sequence, sequence]",
        "causal mask: [sequence, sequence] or [1, 1, sequence, sequence]",
        "attention @ v: [batch, heads, sequence, head_dim]",
        "permute and merge: [batch, sequence, width]",
    ],
}


def recovery_shape_ledger(task):
    return SHAPE_LEDGERS.get(task.template, [TENSOR_CONTRACTS[task.template]])


def task_text(task):
    return (f"# {task.id}\n\nImplement {task.template} using PyTorch, with float32 computations.\n\n"
            + "Config keys: " + ", ".join(task.to_dict()) + ". Use these exact names. "
            "width is the input feature dimension; hidden is the task's hidden/output feature dimension; "
            "length is the nominal sequence/spatial length. Derive actual batch/spatial/sequence sizes from input tensors.\n\n"
            + PARAMETER_CONTRACTS[task.template] + "\n"
            "build(config) constructs the model; parameters are supplied later to load_parameters, not inside config. "
            "Convert loaded NumPy arrays to torch tensors; export_parameters and run must return torch tensors, not NumPy arrays. "
            "Export layout is an interface requirement, not a requirement to change internal storage. "
            "PyTorch transpose swaps two axes; permute reorders multiple axes.\n\n"
            + EQUATIONS[task.template] + "\n\nReturned tensor contract: " + TENSOR_CONTRACTS[task.template] + "\n\nExport dense weights transposed and CNN kernels as OIHW; preserve other parameters. "
            "The public candidate.py docstring specifies the API. Only submission/candidate.py is editable. "
            "The starting candidate.py includes the complete JAX implementation as comments to port. "
            "Verification runs externally; test runners and hidden fixtures are not part of the workspace.\n")
