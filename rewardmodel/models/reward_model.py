from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXConfig, GPTNeoXModel, GPTNeoXPreTrainedModel
from transformers.utils import ModelOutput


class GPTNeoXRewardModelConfig(GPTNeoXConfig):
    model_type = "gpt_neox_reward_model"

    pooling: Literal["mean", "last"]

    def __init__(
        self,
        pooling: Literal["mean", "last"] = "last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pooling = pooling or "last"


@dataclass
class GPTNeoXRewardModelOutput(ModelOutput):
    """
    Reward model output.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Reward score
    """

    logits: torch.FloatTensor = None


class GPTNeoXRewardModel(GPTNeoXPreTrainedModel):
    config_class = GPTNeoXRewardModelConfig

    def __init__(self, config):
        if type(config) == GPTNeoXConfig:
            # When a normal GPTNeoX was loaded it will be converted into a reward model.
            # The direct `type(config) == GPTNeoXConfig` comparison is used (instead of
            # `isinstance()`) since the configuration class of the reward model is also
            # derived form `GPTNeoXConfig`.
            config = GPTNeoXRewardModelConfig.from_dict(config.to_dict())
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.pooling = config.pooling

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> GPTNeoXRewardModelOutput:
        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pooling == "mean":
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        elif self.pooling == "last":
            if attention_mask is None:
                pooled = hidden_states[:, -1]
            else:
                last_idx = attention_mask.cumsum(dim=1).argmax(dim=1)
                pooled = hidden_states.gather(1, last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))).squeeze(
                    1
                )
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        logits = self.out_proj(pooled)

        if not return_dict:
            return (logits,) + outputs[1:]

        return GPTNeoXRewardModelOutput(logits=logits)


AutoConfig.register("gpt_neox_reward_model", GPTNeoXRewardModelConfig)
AutoModelForSequenceClassification.register(GPTNeoXRewardModelConfig, GPTNeoXRewardModel)

class GPTNeoXMORewardModel(GPTNeoXPreTrainedModel):
    config_class = GPTNeoXRewardModelConfig

    def __init__(self, config, n_obj=2, embed_size=256):
        if type(config) == GPTNeoXConfig:
            # When a normal GPTNeoX was loaded it will be converted into a reward model.
            # The direct `type(config) == GPTNeoXConfig` comparison is used (instead of
            # `isinstance()`) since the configuration class of the reward model is also
            # derived form `GPTNeoXConfig`.
            config = GPTNeoXRewardModelConfig.from_dict(config.to_dict())
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.out_proj = nn.Linear(config.hidden_size + embed_size * n_obj, 1)
        self.pooling = config.pooling

        self.objective_embedding = nn.Linear(1, embed_size * n_obj)
        self.objective_weight = nn.Linear(1, n_obj)
        self.n_obj = n_obj
        self.obj_embed_size = embed_size

    def forward(
        self,
        input_ids,
        obj_weight: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> GPTNeoXRewardModelOutput:
        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pooling == "mean":
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        elif self.pooling == "last":
            if attention_mask is None:
                pooled = hidden_states[:, -1]
            else:
                last_idx = attention_mask.cumsum(dim=1).argmax(dim=1)
                pooled = hidden_states.gather(1, last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))).squeeze(
                    1
                )
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # print(f"[func RM forward] pooled.shape: {pooled.shape}") # [N, 2048]
        batch_size = pooled.shape[0]
        unit_input = torch.ones([batch_size, 1]).to(pooled.device)
        # [batch_size, embed_size]
        batch_obj_embed = self.objective_embedding(unit_input)
        # [batch_size, n_obj]
        batch_obj_weight = self.objective_weight(unit_input)

        if obj_weight is None:
            batch_obj_weight = nn.Softmax(dim=1)(batch_obj_weight)
        else:
            batch_obj_weight = obj_weight
        # print(f"batch_obj_embed.shape: {batch_obj_embed.shape}, batch_obj_weight.shape: {batch_obj_weight.shape}")
        # batch_obj_embed.shape: torch.Size([3, 512]), batch_obj_weight.shape: torch.Size([3, 2])
        print(f"{batch_obj_embed.device=}, {batch_obj_weight.device=}")
        for i in range(self.n_obj):
            batch_obj_embed[:, i*self.obj_embed_size:(i+1)*self.obj_embed_size] *= batch_obj_weight[:, i].unsqueeze(1).repeat(1, self.obj_embed_size)

        pooled_cat_embed = torch.cat([pooled, batch_obj_embed], dim=-1)
        logits = self.out_proj(pooled_cat_embed)

        if not return_dict:
            return (logits,) + outputs[1:]

        return GPTNeoXRewardModelOutput(logits=logits)