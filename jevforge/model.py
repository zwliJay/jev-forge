"""JevForge decision model: causal LM backbone + scalar candidate head."""
import torch
from torch import nn


def backbone_hidden_size(config):
    """Return the text hidden size for plain or multimodal HF configs."""
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return hidden_size
    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is not None:
        return hidden_size
    raise ValueError(
        f"Cannot determine text hidden size from {type(config).__name__}"
    )


class JevForgeModel(nn.Module):
    """One forward over padded candidate paths yields one logit per path.

    Grouping the logits of one question's candidates and applying softmax
    produces the complete decision distribution in a single pass — there is
    no token decoding and no LM-head projection.
    """

    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state
        last_index = attention_mask.long().sum(dim=1) - 1
        rows = hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]
        return self.head(rows).squeeze(-1)

    @classmethod
    def from_backbone_config(cls, backbone_config):
        from transformers import AutoModel

        backbone = AutoModel.from_config(backbone_config, attn_implementation="sdpa",
                                         trust_remote_code=False)
        backbone.config.use_cache = False
        return cls(backbone, backbone_hidden_size(backbone_config))
