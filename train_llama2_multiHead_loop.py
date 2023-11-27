import os
import random
import logging
from collections import defaultdict
from typing import Optional, Union, Tuple, List, Callable, Literal
from dataclasses import dataclass

import argparse
import numpy as np
import pandas as pd
from distutils.util import strtobool

import tqdm
import datasets
import accelerate
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model

from transformers import (
    PreTrainedModel,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer
)

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import ModelOutput, is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker, EvalPrediction
from transformers.training_args import OptimizerNames
# class GPTNeoXRewardModelConfig(GPTNeoXConfig):
#     model_type = "gpt_neox_reward_model"

#     pooling: Literal["mean", "last"]

#     def __init__(
#         self,
#         pooling: Literal["mean", "last"] = "last",
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.pooling = pooling or "last"

from utils.llm_utils import get_tokenizer, get_momodel, get_momodel_w, get_momodel_multi_head
from utils.dataset_utils import get_modataset, read_yamls
from utils.sample_utils import PerDatasetSampler
from utils.nn_utils import fuse_gelu, get_loss
from rewardmodel.metrics import RewardMetrics
from dataset.ranking_collator import RankingDataCollator, WRankingDataCollator

class RMTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        sampler: torch.utils.data.sampler.Sampler = None,
        loss_function: Literal["RMLoss"] = "RMLoss",
        score_l2_reg: float = 0.001,
        train_collate_fn: Callable = None,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.train_collate_fn = train_collate_fn
        self.loss_fct = get_loss(loss_function, score_l2_reg=score_l2_reg)
        self.sampler = sampler


    def compute_loss(self, model, inputs, return_logits=False):
        batch, preferences, cu_lens = inputs
        #print(f"{cu_lens=}") # [0, 2]
        #print(f"input_ids.shape: {test_tensor.shape}") # [3, 112]
        #print(f"cu_lens: {cu_lens}") # [0, 3]
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            obj_weight=preferences,
        ).logits
        print(f"{logits=}")

        loss = self.loss_fct(logits, cu_lens)

        return (loss, logits) if return_logits else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[int]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch, preferences, cu_lens = inputs
        with torch.no_grad():
            batch = self._prepare_inputs(batch)
            loss, logits = self.compute_loss(model, (batch, preferences, cu_lens), return_logits=True)

        loss = loss.mean().detach()

        labels = []
        for i, (s, e) in enumerate(zip(cu_lens[:-1], cu_lens[1:])):
            labels.extend([i] * (e - s))
        # make sure labels are same as logits, needed for deepspeed
        labels = torch.tensor(labels, device=logits.device, requires_grad=False).view(-1, 1)
        return (loss, logits.T, labels.T)  # transposed to avoid truncation in evaluation_loop

    def get_train_dataloader(self):
        """
        Inject custom data sampling behaviour into training loop
        and use custom task mixing collate function : train_collate_fn

        rewrite from:
        https://github.com/huggingface/transformers/blob/67d074874d285e616393c65a0e670088e1b6b74a/src/transformers/trainer.py#L846
        """
        data_collator = self.train_collate_fn
        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            # if we are using iterable dataset it means no weight sampling
            # added for backward compat
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self.sampler is None:
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = self.sampler
            logging.warning("Custom sampler found!")

        dataloader = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
        return dataloader

    def get_eval_dataloader(self, train_dataset, collate_fn):
        dataloader = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=collate_fn,
        )
        return dataloader

@dataclass
class GPTNeoXRewardModelOutput(ModelOutput):
    """
    Reward model output.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Reward score
    """

    logits: torch.FloatTensor = None

# https://github.com/huggingface/transformers/blob/f70db28322150dd986298cc1d1be8bc144cc1a88/src/transformers/models/llama/modeling_llama.py#L1147
class LlamaForSequenceClassificationMultiHead(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score1 = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.score2 = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        obj_weight: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits1 = self.score1(hidden_states)
        logits2 = self.score2(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits1.device
                )
            else:
                sequence_lengths = -1

        pooled_logits1 = logits1[torch.arange(batch_size, device=logits1.device), sequence_lengths]
        pooled_logits2 = logits2[torch.arange(batch_size, device=logits2.device), sequence_lengths]
        #print(f"{pooled_logits1=}")
        #print(f"{pooled_logits2=}")

        if obj_weight is None:
            raise NotImplementedError

        n_pair = obj_weight.shape[0]
        batch_obj_weight = torch.cat([obj_weight[i] for i in range(n_pair)], dim=0).to(pooled_logits1.device)

        # unsqueeze(-1).shape == [batch_size * 2, 1]
        logits = batch_obj_weight[:, 0].unsqueeze(-1) * pooled_logits1 + batch_obj_weight[:, 1].unsqueeze(-1) * pooled_logits2
        #print(f"{logits.shape=}") # logits.shape=torch.Size([4, 1])

        #raise NotImplementedError

        # loss = None
        # if labels is not None:
        #     labels = labels.to(logits.device)
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     if self.config.problem_type == "regression":
        #         loss_fct = MSELoss()
        #         if self.num_labels == 1:
        #             loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
        #         else:
        #             loss = loss_fct(pooled_logits, labels)
        #     elif self.config.problem_type == "single_label_classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        #     elif self.config.problem_type == "multi_label_classification":
        #         loss_fct = BCEWithLogitsLoss()
        #         loss = loss_fct(pooled_logits, labels)
        # if not return_dict:
        #     output = (pooled_logits,) + transformer_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutputWithPast(
        #     loss=loss,
        #     logits=pooled_logits,
        #     past_key_values=transformer_outputs.past_key_values,
        #     hidden_states=transformer_outputs.hidden_states,
        #     attentions=transformer_outputs.attentions,
        # )
        return GPTNeoXRewardModelOutput(logits=logits)


#AutoConfig.register("llama2_reward_model", LLAMA2RewardModel)

def batch_w_inference(inputs, model):
    model.eval()
    batch, preference, cu_lens = inputs
    #print(f"{preference=}")
    batch = {k: v.to(model.device) for k, v in batch.items()}
    logits = (
        model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            obj_weight=preference,
        )
        .logits.detach()
        .cpu()
        .numpy()
    )
    print(f"{logits=}")

    labels = []
    for i, (s, e) in enumerate(zip(cu_lens[:-1], cu_lens[1:])):
        labels.extend([i] * (e - s))
    labels = np.array(labels).reshape(-1, 1)
    model.train()
    return EvalPrediction(predictions=logits.T, label_ids=labels.T)

def argument_parsing(notebook=False, notebook_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--no-deepspeed", dest="deepspeed", action="store_false")
    parser.add_argument("--wandb-entity", type=str, default="open-assistant")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from last saved checkpoint")
    parser.add_argument("--rng_seed", type=int, help="rng seed")
    parser.add_argument("--log_wandb", action="store_true", help="whether to report to wandb")
    parser.add_argument("--show_dataset_stats", action="store_true", help="Show dataset stats", default=False)
    parser.set_defaults(deepspeed=False)

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("./configs")
    for name in args.configs:
        if "," in name:
            for n in name.split(","):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    conf["wandb_entity"] = args.wandb_entity
    conf["log_wandb"] = args.log_wandb
    conf["local_rank"] = args.local_rank
    conf["deepspeed"] = args.deepspeed
    conf["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.rng_seed is not None:
        conf["rng_seed"] = args.rng_seed
    conf["show_dataset_stats"] = args.show_dataset_stats

    # get the world size in deepspeed
    if conf["deepspeed"]:
        conf["world_size"] = int(os.getenv("WORLD_SIZE", default="1"))
    else:
        conf["world_size"] = 1

    # Override config from command-line
    def _strtobool(x):
        return bool(strtobool(x))

    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)

    return parser.parse_args(remaining)

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def main():
    training_conf = argument_parsing()
    model_path = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=False).to("cuda:0")
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     load_in_4bit=True,
    # )
    # model.config.use_cache = False

    # add heads and use peft
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    #model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    #model = LlamaForSequenceClassificationMultiHead.from_pretrained(model, num_labels=1, torch_dtype=torch.bfloat16)
    model = LlamaForSequenceClassificationMultiHead.from_pretrained(model_path, num_labels=1, torch_dtype=torch.bfloat16)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    output_dir = (
        training_conf.output_dir
        if training_conf.output_dir
        else f"{training_conf.model_name}-{training_conf.log_dir}-finetuned"
    )

    optimizer = OptimizerNames.ADAMW_BNB if training_conf.quantization else OptimizerNames.ADAMW_HF

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_conf.num_train_epochs,
        warmup_steps=training_conf.warmup_steps,
        learning_rate=float(training_conf.learning_rate),
        deepspeed=training_conf.deepspeed_config if training_conf.deepspeed else None,
        optim=optimizer,
        fp16=training_conf.dtype in ["fp16", "float16"],
        bf16=training_conf.dtype in ["bf16", "bfloat16"],
        local_rank=training_conf.local_rank,
        gradient_checkpointing=training_conf.gradient_checkpointing,
        gradient_accumulation_steps=training_conf.gradient_accumulation_steps,
        per_device_train_batch_size=training_conf.per_device_train_batch_size,
        per_device_eval_batch_size=training_conf.per_device_eval_batch_size,
        adam_beta1=training_conf.adam_beta1,
        adam_beta2=training_conf.adam_beta2,
        adam_epsilon=float(training_conf.adam_epsilon),
        weight_decay=training_conf.weight_decay,
        max_grad_norm=training_conf.max_grad_norm,
        logging_steps=training_conf.logging_steps,
        save_total_limit=training_conf.save_total_limit,
        evaluation_strategy="steps",
        eval_steps=training_conf.eval_steps,
        save_strategy=training_conf.save_strategy,
        save_steps=training_conf.save_steps,
        eval_accumulation_steps=training_conf.eval_accumulation_steps,
        resume_from_checkpoint=training_conf.resume_from_checkpoint,
        report_to="wandb" if training_conf.log_wandb else None,
    )

    wh_train, w_train, wh_evals, w_evals = get_modataset(training_conf, mode="rm")

    w_train_collate_fn = WRankingDataCollator(
        tokenizer,
        max_length=training_conf.max_length,
        pad_to_multiple_of=16,
        max_replies=training_conf.max_replies,
    )
    w_eval_collate_fn = WRankingDataCollator(
        tokenizer,
        max_length=training_conf.max_length,
        pad_to_multiple_of=16,
        max_replies=training_conf.max_replies,
    )

    show_dataset_stats = (training_conf.verbose or training_conf.show_dataset_stats) and (
        not training_conf.deepspeed or training_conf.local_rank == 0
    )
    if show_dataset_stats:
        print("Dataset stats before sampling:")
        total = len(wh_train)
        for d in wh_train.datasets:
            if isinstance(d, Subset):
                name = f"Subset of {type(d.dataset).__name__}"
                if hasattr(d.dataset, "name"):
                    name += f" ({d.dataset.name})"
            else:
                name = type(d).__name__
                if hasattr(d, "name"):
                    name += f" ({d.name})"
            print(f"{name}: {len(d)} ({len(d) / total:%})")
        print(f"Total train: {total}")

    if training_conf.use_custom_sampler:
        w_samples_length = None
        if training_conf.sort_by_length:
            w_samples_length = list(
                map(
                    lambda x: w_train_collate_fn.process_one(x, return_length=True),
                    tqdm(w_train, desc="Calculating lengths per sample"),
                )
            )

        w_sampler = PerDatasetSampler.build_w_sampler_from_config(
            training_conf,
            w_train.datasets,
            rank=training_conf.local_rank,
            world_size=training_conf.world_size,
            samples_length=w_samples_length,
            verbose=show_dataset_stats,
        )
    else:
        w_sampler = None

    optimizer = AdamW(model.parameters(), lr=float(training_conf.learning_rate), weight_decay=float(training_conf.weight_decay))

    compute_metrics = RewardMetrics(training_conf.metrics)
    trainer = RMTrainer(
        model=model,
        args=args,
        sampler=w_sampler,
        train_collate_fn=w_train_collate_fn,
        loss_function=training_conf.loss_fn,
        score_l2_reg=training_conf.score_l2_reg,
        train_dataset=w_train,
        eval_dataset=w_evals,
        data_collator=w_eval_collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    w_eval_dataloaders = {k : trainer.get_eval_dataloader(w_eval, w_eval_collate_fn) for (k, w_eval) in w_evals.items()}
    for dataset_name, w_eval in w_eval_dataloaders.items():
        score_dict = defaultdict(float)

        for tmp_id, data in enumerate(w_eval):
            batch, preference, cu_lens = data
            print(f'{batch["input_ids"]}')
            raise NotImplementedError
            eval_pred = batch_w_inference(data, model)
            results = compute_metrics(eval_pred)
            for metric in training_conf.metrics:
                score_dict[metric] += results.get(metric)

        score_dict = {k: round(v / len(w_eval), 3) for k, v in score_dict.items()}
        print(score_dict)

    trainer.train(resume_from_checkpoint=training_conf.resume_from_checkpoint)


if __name__ == "__main__":
    main()