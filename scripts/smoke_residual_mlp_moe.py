import tempfile
from pathlib import Path

import torch
from torch import nn

from aspect_residual_mlp_moe import (
    inject_aspect_residual_mlp_moe,
    load_mlp_moe_adapter,
    residual_mlp_trainable_parameters,
    save_mlp_moe_adapter,
    set_mlp_moe_gates,
)


class Config:
    hidden_size = 16


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(Config.hidden_size, Config.hidden_size)

    def forward(self, hidden_states):
        return (self.proj(hidden_states),)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([FakeLayer() for _ in range(4)])
        self.head = nn.Linear(Config.hidden_size, 7)

    def forward(self, hidden_states):
        x = hidden_states
        for layer in self.model.language_model.layers:
            x = layer(x)[0]
        return self.head(x)


def main():
    model = FakeModel()
    for p in model.parameters():
        p.requires_grad_(False)
    injected = inject_aspect_residual_mlp_moe(
        model,
        num_experts=6,
        bottleneck_size=4,
        residual_scale=0.1,
        layers="last_2",
    )
    params = residual_mlp_trainable_parameters(model)
    assert injected == ["model.language_model.layers.2", "model.language_model.layers.3"]
    assert params and sum(p.numel() for p in params) < sum(p.numel() for p in model.parameters())

    x = torch.randn(1, 3, Config.hidden_size)
    gates = torch.tensor([[0.1, 0.2, 0.3, 0.1, 0.2, 0.1]])
    set_mlp_moe_gates(model, gates)
    logits = model(x)
    assert torch.isfinite(logits).all()

    config = {
        "moe_impl": "residual_mlp_moe",
        "num_experts": 6,
        "bottleneck_size": 4,
        "dropout": 0.05,
        "residual_scale": 0.1,
        "layers": "last_2",
        "use_shared": True,
        "activation": "silu",
        "zero_init": True,
    }
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        save_mlp_moe_adapter(model, None, out, config)
        assert (out / "mlp_moe_adapter.pt").exists()
        assert (out / "mlp_moe_config.json").exists()

        reloaded = FakeModel()
        for p in reloaded.parameters():
            p.requires_grad_(False)
        inject_aspect_residual_mlp_moe(reloaded, num_experts=6, bottleneck_size=4, layers="last_2")
        load_mlp_moe_adapter(reloaded, out)
        set_mlp_moe_gates(reloaded, gates)
        assert torch.isfinite(reloaded(x)).all()

    print("residual_mlp_moe smoke ok")


if __name__ == "__main__":
    main()
