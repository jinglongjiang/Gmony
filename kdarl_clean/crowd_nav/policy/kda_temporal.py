"""Official FLA KDA layers in the inherited residual temporal interface."""

from torch import nn


class KDATemporalEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        from fla.layers.kda import KimiDeltaAttention

        if d_model % 64:
            raise ValueError("KDA width must be divisible by the fixed head dimension 64")
        self.backbone_name = 'kda'
        self.n_layers = n_layers
        self.backend = nn.ModuleList([
            nn.ModuleList([
                KimiDeltaAttention(hidden_size=d_model, head_dim=64,
                                   num_heads=d_model // 64, layer_idx=i),
                nn.LayerNorm(d_model),
            ]) for i in range(n_layers)
        ])

    def forward(self, x, mask=None):
        if mask is not None:
            raise ValueError("VL-v2 uses fixed, left-repeated windows; mask must be None")
        if x.ndim != 3:
            raise ValueError("Expected [batch, time, width]")
        if not x.is_cuda:
            raise RuntimeError("Official optimized KDA requires CUDA; no substitute backend")
        for layer, norm in self.backend:
            y, _, cache = layer(x, past_key_values=None, use_cache=False)
            if cache is not None:
                raise RuntimeError("KDA unexpectedly returned persistent window state")
            x = x + norm(y)
        return x
