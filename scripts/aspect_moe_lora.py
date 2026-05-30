import json
import math
import re
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


DIMS = ["overall_quality", "empathy", "specificity", "medical_advice", "factual_consistency", "toxicity"]


def normalize_gates(g, eps=1e-4):
    if not torch.is_tensor(g):
        g = torch.tensor(g, dtype=torch.float32)
    if g.dim() == 1:
        g = g.unsqueeze(0)
    g = g.float().clamp_min(0.0)
    tau = g + float(eps)
    return tau / tau.sum(dim=-1, keepdim=True).clamp_min(float(eps))


class AspectMoELinear(nn.Module):
    def __init__(
        self,
        base_linear,
        num_experts=6,
        r_shared=8,
        r_expert=8,
        alpha_shared=16,
        alpha_expert=16,
        dropout=0.05,
        moe_eps=1e-4,
        lora_dtype=torch.float32,
    ):
        super().__init__()
        if not hasattr(base_linear, "in_features") or not hasattr(base_linear, "out_features"):
            raise TypeError(f"Cannot wrap module without in_features/out_features: {type(base_linear)}")

        self.base_linear = base_linear
        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)
        self.num_experts = int(num_experts)
        self.r_shared = int(r_shared)
        self.r_expert = int(r_expert)
        self.alpha_shared = float(alpha_shared)
        self.alpha_expert = float(alpha_expert)
        self.shared_scale = self.alpha_shared / max(1, self.r_shared)
        self.expert_scale = self.alpha_expert / max(1, self.r_expert)
        self.moe_eps = float(moe_eps)
        self.dropout = nn.Dropout(float(dropout))
        self.current_gates = None

        device = self._infer_device(base_linear)
        self.A_shared = nn.Parameter(torch.empty(self.r_shared, self.in_features, device=device, dtype=lora_dtype))
        self.B_shared = nn.Parameter(torch.empty(self.out_features, self.r_shared, device=device, dtype=lora_dtype))
        self.A_expert = nn.Parameter(
            torch.empty(self.num_experts, self.r_expert, self.in_features, device=device, dtype=lora_dtype)
        )
        self.B_expert = nn.Parameter(
            torch.empty(self.num_experts, self.out_features, self.r_expert, device=device, dtype=lora_dtype)
        )

        self.reset_parameters()
        for p in self.base_linear.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _infer_device(module):
        for p in module.parameters(recurse=True):
            return p.device
        return torch.device("cpu")

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A_shared, a=math.sqrt(5))
        nn.init.zeros_(self.B_shared)
        for k in range(self.num_experts):
            nn.init.kaiming_uniform_(self.A_expert[k], a=math.sqrt(5))
            nn.init.zeros_(self.B_expert[k])

    def set_gates(self, g):
        self.current_gates = g.detach()

    def _tau_for_input(self, x):
        if self.current_gates is None:
            tau = torch.full(
                (1, self.num_experts),
                1.0 / self.num_experts,
                device=x.device,
                dtype=self.A_shared.dtype,
            )
        else:
            tau = normalize_gates(self.current_gates, self.moe_eps).to(device=x.device, dtype=self.A_shared.dtype)

        if x.dim() == 2:
            n = x.shape[0]
            if tau.shape[0] == n:
                return tau
            if tau.shape[0] == 1:
                return tau.expand(n, -1)
            if n % tau.shape[0] == 0:
                return tau.repeat_interleave(n // tau.shape[0], dim=0)
            return tau.mean(dim=0, keepdim=True).expand(n, -1)

        batch = x.shape[0]
        if tau.shape[0] == batch:
            return tau
        if tau.shape[0] == 1:
            return tau.expand(batch, -1)
        return tau.mean(dim=0, keepdim=True).expand(batch, -1)

    def _tau_view(self, tau, x):
        if x.dim() == 2:
            return (tau.shape[0], 1)
        return (tau.shape[0],) + (1,) * (x.dim() - 2) + (1,)

    def forward(self, x):
        base = self.base_linear(x)
        lora_x = self.dropout(x).to(dtype=self.A_shared.dtype)

        shared_down = F.linear(lora_x, self.A_shared)
        shared = F.linear(shared_down, self.B_shared) * self.shared_scale

        tau = self._tau_for_input(x)
        expert_mix = torch.zeros_like(shared)
        tau_shape = self._tau_view(tau, x)
        for k in range(self.num_experts):
            expert_down = F.linear(lora_x, self.A_expert[k])
            expert_up = F.linear(expert_down, self.B_expert[k]) * self.expert_scale
            expert_mix = expert_mix + expert_up * tau[:, k].reshape(tau_shape)

        return base + (shared + expert_mix).to(dtype=base.dtype)


def _module_parent(model, module_name):
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _is_linear_like(module):
    return (
        not isinstance(module, AspectMoELinear)
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
        and callable(module)
    )


def freeze_model_parameters(model):
    for p in model.parameters():
        p.requires_grad_(False)


def wrap_aspect_moe_layers(model, config):
    target_regex = config["target_regex"]
    pattern = re.compile(target_regex)
    wrapped = []

    for name, module in list(model.named_modules()):
        if not name:
            continue
        if not pattern.fullmatch(name):
            continue
        if not _is_linear_like(module):
            print(f"[skip] target matched but is not linear-like: {name} ({type(module).__name__})")
            continue

        parent, child_name = _module_parent(model, name)
        wrapped_module = AspectMoELinear(
            module,
            num_experts=len(config.get("dims", DIMS)),
            r_shared=config["r_shared"],
            r_expert=config["r_expert"],
            alpha_shared=config["alpha_shared"],
            alpha_expert=config["alpha_expert"],
            dropout=config["dropout"],
            moe_eps=config.get("moe_eps", 1e-4),
        )
        setattr(parent, child_name, wrapped_module)
        wrapped.append(name)

    if not wrapped:
        raise ValueError(f"No modules were wrapped with target_regex={target_regex!r}")
    return wrapped


def set_moe_gates(model, g):
    n = 0
    for module in model.modules():
        if isinstance(module, AspectMoELinear):
            module.set_gates(g)
            n += 1
    if n == 0:
        raise RuntimeError("No AspectMoELinear modules found. Did you call wrap_aspect_moe_layers?")


def moe_adapter_state_dict(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, AspectMoELinear):
            prefix = f"{name}."
            state[prefix + "A_shared"] = module.A_shared.detach().cpu()
            state[prefix + "B_shared"] = module.B_shared.detach().cpu()
            state[prefix + "A_expert"] = module.A_expert.detach().cpu()
            state[prefix + "B_expert"] = module.B_expert.detach().cpu()
    return state


def save_moe_adapter(model, tokenizer, output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(moe_adapter_state_dict(model), output_dir / "moe_adapter.pt")
    with open(output_dir / "moe_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    tokenizer.save_pretrained(output_dir)
    print("[save] moe adapter ->", output_dir)


def load_moe_config(adapter_dir):
    with open(Path(adapter_dir) / "moe_config.json", encoding="utf-8") as f:
        return json.load(f)


def load_moe_adapter(model, adapter_dir, map_location="cpu"):
    path = Path(adapter_dir) / "moe_adapter.pt"
    state = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading MoE adapter: {unexpected[:20]}")
    loaded = len(state)
    missing_moe = [k for k in missing if any(x in k for x in ("A_shared", "B_shared", "A_expert", "B_expert"))]
    if missing_moe:
        raise RuntimeError(f"Missing MoE adapter keys: {missing_moe[:20]}")
    print(f"[load] moe adapter <- {path} ({loaded} tensors)")
    return loaded


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(1, total)
    return trainable, total, pct
