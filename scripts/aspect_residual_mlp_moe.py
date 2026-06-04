import json
import re
from pathlib import Path

import torch
from torch import nn

from aspect_moe_lora import DIMS, normalize_gates


def activation_module(name):
    name = str(name or "silu").lower()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported residual MLP MoE activation: {name}")


class BottleneckMLP(nn.Module):
    def __init__(self, hidden_size, bottleneck_size, dropout=0.05, activation="silu", zero_init=True):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.act = activation_module(activation)
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(bottleneck_size, hidden_size, bias=False)
        if zero_init:
            nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.dropout(self.act(self.down(x))))


class AspectResidualMLPMoE(nn.Module):
    """Aspect-conditioned residual MLP experts.

    Implements:
        h_out = h + residual_scale * (shared_mlp(norm(h)) + sum_k tau_k(g) * expert_mlp_k(norm(h)))
    """

    def __init__(
        self,
        hidden_size,
        num_experts=6,
        bottleneck_size=64,
        dropout=0.05,
        residual_scale=0.1,
        activation="silu",
        use_shared=True,
        zero_init=True,
        moe_eps=1e-4,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.bottleneck_size = int(bottleneck_size)
        self.residual_scale = float(residual_scale)
        self.use_shared = bool(use_shared)
        self.activation = str(activation)
        self.zero_init = bool(zero_init)
        self.moe_eps = float(moe_eps)
        self.current_gates = None

        self.norm = nn.LayerNorm(self.hidden_size)
        self.shared_mlp = (
            BottleneckMLP(self.hidden_size, self.bottleneck_size, dropout, activation, zero_init)
            if self.use_shared
            else None
        )
        self.expert_mlps = nn.ModuleList(
            [
                BottleneckMLP(self.hidden_size, self.bottleneck_size, dropout, activation, zero_init)
                for _ in range(self.num_experts)
            ]
        )

    def set_gates(self, gates, detach=True):
        if gates is None:
            self.current_gates = None
        else:
            self.current_gates = gates.detach() if detach else gates

    def clear_gates(self):
        self.current_gates = None

    def _tau_for_hidden(self, h):
        batch = h.shape[0]
        if self.current_gates is None:
            return torch.full(
                (batch, self.num_experts),
                1.0 / self.num_experts,
                device=h.device,
                dtype=torch.float32,
            )

        tau = normalize_gates(self.current_gates, self.moe_eps).to(device=h.device, dtype=torch.float32)
        if tau.shape[-1] < self.num_experts:
            pad = torch.zeros(tau.shape[0], self.num_experts - tau.shape[-1], device=tau.device, dtype=tau.dtype)
            tau = torch.cat([tau, pad], dim=-1)
            tau = normalize_gates(tau, self.moe_eps)
        elif tau.shape[-1] > self.num_experts:
            tau = normalize_gates(tau[:, : self.num_experts], self.moe_eps)

        if tau.shape[0] == batch:
            return tau
        if tau.shape[0] == 1:
            return tau.expand(batch, -1)
        raise ValueError(
            f"Incompatible residual MLP MoE gates batch size {tau.shape[0]} for hidden batch size {batch}"
        )

    def forward(self, h):
        if h.dim() != 3:
            raise ValueError(f"AspectResidualMLPMoE expects [batch, seq, hidden], got shape {tuple(h.shape)}")
        x = self.norm(h.float())
        delta = torch.zeros_like(x)
        if self.shared_mlp is not None:
            delta = delta + self.shared_mlp(x)

        tau = self._tau_for_hidden(h)
        for k, expert in enumerate(self.expert_mlps):
            delta = delta + expert(x) * tau[:, k].view(-1, 1, 1)
        return h + (self.residual_scale * delta).to(dtype=h.dtype)


class ResidualMLPMoELayerWrapper(nn.Module):
    def __init__(self, layer, adapter):
        super().__init__()
        self.layer = layer
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        if torch.is_tensor(out):
            return self.adapter(out)
        if isinstance(out, tuple):
            if not out:
                return out
            return (self.adapter(out[0]),) + out[1:]
        if isinstance(out, list):
            if not out:
                return out
            out = list(out)
            out[0] = self.adapter(out[0])
            return out
        raise TypeError(f"Unsupported decoder layer output type for residual_mlp_moe: {type(out)}")

    def set_gates(self, gates, detach=True):
        self.adapter.set_gates(gates, detach=detach)

    def clear_gates(self):
        self.adapter.clear_gates()


def _get_attr_path(root, path):
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _find_decoder_layers(model):
    candidates = [
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "layers",
    ]
    for path in candidates:
        try:
            layers = _get_attr_path(model, path)
        except AttributeError:
            continue
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return path, layers
    raise RuntimeError("Could not find Gemma decoder layers for residual_mlp_moe injection")


def _infer_hidden_size(model, layers):
    for attr in ["hidden_size", "text_config.hidden_size"]:
        try:
            value = _get_attr_path(model.config, attr)
        except AttributeError:
            continue
        if value:
            return int(value)

    for layer in layers:
        for module in layer.modules():
            if hasattr(module, "in_features"):
                return int(module.in_features)
    raise RuntimeError("Could not infer hidden size for residual_mlp_moe")


def _infer_module_device(module):
    for p in module.parameters(recurse=True):
        return p.device
    return torch.device("cpu")


def parse_layer_spec(spec, n_layers):
    spec = str(spec or "last_8").strip().lower()
    if spec == "all":
        return list(range(n_layers))
    m = re.fullmatch(r"last_(\d+)", spec)
    if m:
        n = min(n_layers, int(m.group(1)))
        return list(range(n_layers - n, n_layers))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx = n_layers + idx
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer index {part} out of range for {n_layers} layers")
        out.append(idx)
    if not out:
        raise ValueError(f"No residual_mlp_moe layers selected from spec={spec!r}")
    return sorted(set(out))


def inject_aspect_residual_mlp_moe(
    model,
    num_experts=6,
    bottleneck_size=64,
    dropout=0.05,
    residual_scale=0.1,
    layers="last_8",
    use_shared=True,
    activation="silu",
    zero_init=True,
    moe_eps=1e-4,
):
    layer_path, decoder_layers = _find_decoder_layers(model)
    hidden_size = _infer_hidden_size(model, decoder_layers)
    indices = parse_layer_spec(layers, len(decoder_layers))
    wrapped = []
    for idx in indices:
        if isinstance(decoder_layers[idx], ResidualMLPMoELayerWrapper):
            wrapped.append(f"{layer_path}.{idx}")
            continue
        adapter = AspectResidualMLPMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            bottleneck_size=bottleneck_size,
            dropout=dropout,
            residual_scale=residual_scale,
            activation=activation,
            use_shared=use_shared,
            zero_init=zero_init,
            moe_eps=moe_eps,
        )
        adapter.to(device=_infer_module_device(decoder_layers[idx]))
        decoder_layers[idx] = ResidualMLPMoELayerWrapper(decoder_layers[idx], adapter)
        wrapped.append(f"{layer_path}.{idx}")
    return wrapped


def set_mlp_moe_gates(model, gates, detach=True):
    n = 0
    for module in model.modules():
        if isinstance(module, AspectResidualMLPMoE):
            module.set_gates(gates, detach=detach)
            n += 1
    if n == 0:
        raise RuntimeError("No AspectResidualMLPMoE modules found. Did you inject residual_mlp_moe adapters?")


def clear_mlp_moe_gates(model):
    for module in model.modules():
        if isinstance(module, AspectResidualMLPMoE):
            module.clear_gates()


def mlp_moe_adapter_state_dict(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, AspectResidualMLPMoE):
            for key, value in module.state_dict().items():
                state[f"{name}.{key}"] = value.detach().cpu()
    return state


def save_mlp_moe_adapter(model, tokenizer, output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mlp_moe_adapter_state_dict(model), output_dir / "mlp_moe_adapter.pt")
    with open(output_dir / "mlp_moe_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    print("[save] residual_mlp_moe adapter ->", output_dir)


def load_mlp_moe_config(adapter_dir):
    with open(Path(adapter_dir) / "mlp_moe_config.json", encoding="utf-8") as f:
        return json.load(f)


def load_mlp_moe_adapter(model, adapter_dir, map_location="cpu"):
    path = Path(adapter_dir) / "mlp_moe_adapter.pt"
    state = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading residual_mlp_moe adapter: {unexpected[:20]}")
    missing_mlp = [k for k in missing if ".adapter." in k and ("expert_mlps" in k or "shared_mlp" in k or "norm" in k)]
    if missing_mlp:
        raise RuntimeError(f"Missing residual_mlp_moe adapter keys: {missing_mlp[:20]}")
    print(f"[load] residual_mlp_moe adapter <- {path} ({len(state)} tensors)")
    return len(state)


def residual_mlp_trainable_parameters(model):
    params = []
    for module in model.modules():
        if isinstance(module, AspectResidualMLPMoE):
            params.extend(list(module.parameters()))
    return [p for p in params if p.requires_grad]


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(1, total)
    return trainable, total, pct
