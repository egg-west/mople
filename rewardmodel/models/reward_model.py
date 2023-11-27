from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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

@dataclass
class GPTNeoXRewardModelOutputAlternative(ModelOutput):
    """
    Reward model output.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Reward score
    """

    logits: torch.FloatTensor = None
    alternative: torch.FloatTensor = None


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
    """add learnable objective embedding, concatenated with h from LLM"""
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
            batch_obj_weight = obj_weight.to(pooled.device)
        # print(f"batch_obj_embed.shape: {batch_obj_embed.shape}, batch_obj_weight.shape: {batch_obj_weight.shape}")
        # batch_obj_embed.shape: torch.Size([3, 512]), batch_obj_weight.shape: torch.Size([3, 2])
        # print(f"{batch_obj_embed.device=}, {batch_obj_weight.device=}") # is on cpu
        for i in range(self.n_obj):
            batch_obj_embed[:, i*self.obj_embed_size:(i+1)*self.obj_embed_size] *= batch_obj_weight[:, i].unsqueeze(1).repeat(1, self.obj_embed_size)

        pooled_cat_embed = torch.cat([pooled, batch_obj_embed], dim=-1)
        logits = self.out_proj(pooled_cat_embed)

        if not return_dict:
            return (logits,) + outputs[1:]

        return GPTNeoXRewardModelOutput(logits=logits)

class GPTNeoXMORewardModelMultiHead(GPTNeoXPreTrainedModel):
    """A multi-head reward model, each head predict a single-objective reward.
    e.g. helpful reward"""
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
        self.out_proj_task0 = nn.Linear(config.hidden_size, 1)
        self.out_proj_task1 = nn.Linear(config.hidden_size, 1)
        self.pooling = config.pooling

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

        task0_logits = self.out_proj_task0(pooled)
        task1_logits = self.out_proj_task1(pooled)
        if obj_weight is None:
            raise NotImplementedError

        n_pair = obj_weight.shape[0]
        batch_obj_weight = torch.cat([obj_weight[i] for i in range(n_pair)], dim=0).to(pooled.device)

        # unsqueeze(-1).shape == [batch_size * 2, 1]
        logits = batch_obj_weight[:, 0].unsqueeze(-1) * task0_logits + batch_obj_weight[:, 1].unsqueeze(-1) * task1_logits

        if not return_dict:
            return (logits,) + outputs[1:]

        return GPTNeoXRewardModelOutput(logits=logits)

class GPTNeoXMORewardModelMultiHeadPref(GPTNeoXPreTrainedModel):
    """A multi-head reward model, each head predict a single-objective reward.
    e.g. helpful reward"""
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
        self.out_proj_preference = nn.Linear(config.hidden_size, 2)
        self.out_proj_task0 = nn.Linear(config.hidden_size, 1)
        self.out_proj_task1 = nn.Linear(config.hidden_size, 1)
        self.pooling = config.pooling

        self.pref_softmax = nn.Softmax(dim=1)

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
    ) -> GPTNeoXRewardModelOutputAlternative:
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

        task0_logits = self.out_proj_task0(pooled)
        task1_logits = self.out_proj_task1(pooled)
        preferences = self.pref_softmax(self.out_proj_preference(pooled))
        if obj_weight is None:
            # print(f"{task0_logits.shape=}, {preferences.shape=}")
            # task0_logits.shape=torch.Size([bs, 1]), preferences.shape=torch.Size([bs, 2])

            # use preference to weight the reward
            cat_rewards = torch.cat([task0_logits, task1_logits], dim=1)
            #print(f"{cat_rewards.shape=}") # [bs, 2]

            logits = preferences * cat_rewards
            if not return_dict:
                return (logits,) + outputs[1:]

            return GPTNeoXRewardModelOutputAlternative(logits=logits, alternative=None)

        n_pair = obj_weight.shape[0]
        batch_obj_weight = torch.cat([obj_weight[i] for i in range(n_pair)], dim=0).to(pooled.device)
        # print(f"{batch_obj_weight.shape=}")# torch.Size([4, 2])

        alternative = ((preferences - batch_obj_weight)**2).mean()
        # unsqueeze(-1).shape == [batch_size * 2, 1]
        logits = batch_obj_weight[:, 0].unsqueeze(-1) * task0_logits + batch_obj_weight[:, 1].unsqueeze(-1) * task1_logits

        if not return_dict:
            return (logits,) + outputs[1:]

        return GPTNeoXRewardModelOutputAlternative(logits=logits, alternative=alternative)

class GPTNeoXRewardModelVarianceOutput(ModelOutput):
    """
    Reward model output.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Reward score
    """

    logits: torch.FloatTensor = None
    var: torch.FloatTensor = None

class GPTNeoXMORewardModelMultiHeadVariance(GPTNeoXPreTrainedModel):
    """add learnable objective embedding, concatenated with h from LLM
    maximize the variance between mean value of latent feature for each task"""
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
        self.out_proj_task0 = nn.Linear(config.hidden_size, 1)
        self.out_proj_task1 = nn.Linear(config.hidden_size, 1)
        self.pooling = config.pooling

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

        task0_logits = self.out_proj_task0(pooled)
        task1_logits = self.out_proj_task1(pooled)
        if obj_weight is None:
            raise NotImplementedError

        n_pair = obj_weight.shape[0]
        batch_obj_weight = torch.cat([obj_weight[i] for i in range(n_pair)], dim=0).to(pooled.device)

        # unsqueeze(-1).shape == [batch_size * 2, 1]
        logits = batch_obj_weight[:, 0].unsqueeze(-1) * task0_logits + batch_obj_weight[:, 1].unsqueeze(-1) * task1_logits

        # my variance loss
        if batch_obj_weight[:, 0].any() and batch_obj_weight[:, 0].any():
            mean_0 = (batch_obj_weight[:, 0].unsqueeze(-1) * pooled).mean(dim=0)
            mean_1 = (batch_obj_weight[:, 1].unsqueeze(-1) * pooled).mean(dim=0)
            mean_mean = (mean_0 + mean_1) / 2.0
            variance = ((mean_0 - mean_mean)**2).mean() + ((mean_1 - mean_mean)**2).mean()
        else:
            variance = None
        if not return_dict:
            return (logits,) + outputs[1:]

        return GPTNeoXRewardModelVarianceOutput(logits=logits, var=variance)

class GPTNeoXMORewardModel_W(GPTNeoXPreTrainedModel):
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
        self.out_proj0 = nn.Linear(config.hidden_size + embed_size * n_obj, config.hidden_size)
        self.out_proj1 = nn.Linear(config.hidden_size, 1)
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
            n_pair = obj_weight.shape[0]
            batch_obj_weight = torch.cat([obj_weight[i] for i in range(n_pair)], dim=0).to(pooled.device)
            #batch_obj_weight = obj_weight.to(pooled.device)
        # print(f"batch_obj_embed.shape: {batch_obj_embed.shape}, batch_obj_weight.shape: {batch_obj_weight.shape}")
        # batch_obj_embed.shape: torch.Size([3, 512]), batch_obj_weight.shape: torch.Size([3, 2])
        #print(f"{batch_obj_weight}=")
        for i in range(self.n_obj):
            #print(f"{batch_obj_embed[:, i*self.obj_embed_size:(i+1)*self.obj_embed_size].shape=}, {batch_obj_weight[:, i].unsqueeze(1).repeat(1, self.obj_embed_size).shape=}")
            #batch_obj_embed[:, i*self.obj_embed_size:(i+1)*self.obj_embed_size] *= batch_obj_weight[:, i].unsqueeze(1).repeat(1, self.obj_embed_size)
            batch_obj_embed[:, i*self.obj_embed_size:(i+1)*self.obj_embed_size] *= batch_obj_weight[:, i].unsqueeze(1).repeat(1, self.obj_embed_size)

        pooled_cat_embed = torch.cat([pooled, batch_obj_embed], dim=-1)
        logits = self.out_proj1(F.relu(self.out_proj0(pooled_cat_embed)))

        if not return_dict:
            return (logits,) + outputs[1:]

        return GPTNeoXRewardModelOutput(logits=logits)

@dataclass
class GPTNeoXRewardModelConservativeOutput(ModelOutput):
    """
    Reward model output.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Reward score
    """

    logits: torch.FloatTensor = None
    obj_weights: torch.FloatTensor = None

class GPTNeoXMORewardModel_Conservative(GPTNeoXPreTrainedModel):
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
        self.out_proj0 = nn.Linear(config.hidden_size + embed_size * n_obj, config.hidden_size)
        self.out_proj1 = nn.Linear(config.hidden_size, 1)
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
            n_pair = obj_weight.shape[0]
            batch_obj_weight = torch.cat([obj_weight[i] for i in range(n_pair)], dim=0).to(pooled.device)
            #batch_obj_weight = obj_weight.to(pooled.device)
        # print(f"batch_obj_embed.shape: {batch_obj_embed.shape}, batch_obj_weight.shape: {batch_obj_weight.shape}")
        # batch_obj_embed.shape: torch.Size([3, 512]), batch_obj_weight.shape: torch.Size([3, 2])
        #print(f"{batch_obj_weight}=")
        for i in range(self.n_obj):
            #print(f"{batch_obj_embed[:, i*self.obj_embed_size:(i+1)*self.obj_embed_size].shape=}, {batch_obj_weight[:, i].unsqueeze(1).repeat(1, self.obj_embed_size).shape=}")
            batch_obj_embed[:, i*self.obj_embed_size:(i+1)*self.obj_embed_size] *= batch_obj_weight[:, i].unsqueeze(1).repeat(1, self.obj_embed_size)

        pooled_cat_embed = torch.cat([pooled, batch_obj_embed], dim=-1)
        logits = self.out_proj1(F.relu(self.out_proj0(pooled_cat_embed)))

        if not return_dict:
            return (logits,) + outputs[1:]

        #print(f"{batch_obj_weight}=, {logits=}")
        #batch_obj_weight[0, :] * logits

        return GPTNeoXRewardModelConservativeOutput(logits=logits, obj_weights=batch_obj_weight)