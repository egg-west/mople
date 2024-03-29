import argparse
from collections import defaultdict
import logging
import os
from distutils.util import strtobool
from typing import Callable, Literal, Optional, Union

import datasets
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
from transformers import PreTrainedModel, Trainer, TrainingArguments, get_scheduler
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker, EvalPrediction
from transformers.training_args import OptimizerNames
from transformers.utils import is_datasets_available

from utils.llm_utils import get_tokenizer, get_momodel
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
        batch, cu_lens = inputs
        #print(f"input_ids.shape: {test_tensor.shape}") # [3, 112]
        #print(f"cu_lens: {cu_lens}") # [0, 3]
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        loss = self.loss_fct(logits, cu_lens)

        return (loss, logits) if return_logits else loss

    def compute_w_loss(self, model, inputs, return_logits=False):
        batch, preference, cu_lens = inputs
        #print(f"input_ids.shape: {test_tensor.shape}") # [3, 112]
        #print(f"cu_lens: {cu_lens}") # [0, 3]
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            obj_weight=preference,
        ).logits

        loss = self.loss_fct(logits, cu_lens)

        return (loss, logits) if return_logits else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[int]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch, cu_lens = inputs
        with torch.no_grad():
            batch = self._prepare_inputs(batch)
            loss, logits = self.compute_loss(model, (batch, cu_lens), return_logits=True)

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

    def get_w_train_dataloader(self, train_dataset, collate_fn, sampler):
        """
        Inject custom data sampling behaviour into training loop
        and use custom task mixing collate function : train_collate_fn

        rewrite from:
        https://github.com/huggingface/transformers/blob/67d074874d285e616393c65a0e670088e1b6b74a/src/transformers/trainer.py#L846
        """
        data_collator = collate_fn
        train_dataset = train_dataset
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
        """
        if self.sampler is None:
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = self.sampler
            logging.warning("Custom sampler found!")
        """
        train_sampler = sampler
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

def batch_inference(inputs, model):
    model.eval()
    batch, cu_lens = inputs
    batch = {k: v.to(model.device) for k, v in batch.items()}
    logits = (
        model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        .logits.detach()
        .cpu()
        .numpy()
    )

    labels = []
    for i, (s, e) in enumerate(zip(cu_lens[:-1], cu_lens[1:])):
        labels.extend([i] * (e - s))
    labels = np.array(labels).reshape(-1, 1)
    model.train()
    return EvalPrediction(predictions=logits.T, label_ids=labels.T)

def batch_w_inference(inputs, model):
    model.eval()
    batch, preference, cu_lens = inputs
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

def main():
    training_conf = argument_parsing()
    tokenizer = get_tokenizer(training_conf)
    model = get_momodel(training_conf, tokenizer)
    # test the data loader

    wh_train, w_train, wh_evals, w_evals = get_modataset(training_conf, mode="rm")

    train_collate_fn = RankingDataCollator(
        tokenizer,
        max_length=training_conf.max_length,
        pad_to_multiple_of=16,
        max_replies=training_conf.max_replies,
        #use_system_tag=training_conf.use_system_tag,
        #system_property_dropout=training_conf.system_property_dropout,
        #system_add_length=training_conf.system_add_length,
    )
    eval_collate_fn = RankingDataCollator(
        tokenizer,
        max_length=training_conf.max_length,
        pad_to_multiple_of=16,
        max_replies=training_conf.max_replies,
        #use_system_tag=training_conf.use_system_tag,
        #system_property_dropout=training_conf.system_property_dropout,
        #system_add_length=training_conf.system_add_length,
    )
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
        samples_length = None
        w_samples_length = None
        if training_conf.sort_by_length:
            samples_length = list(
                map(
                    lambda x: train_collate_fn.process_one(x, return_length=True),
                    tqdm(wh_train, desc="Calculating lengths per sample"),
                )
            )
            w_samples_length = list(
                map(
                    lambda x: w_train_collate_fn.process_one(x, return_length=True),
                    tqdm(w_train, desc="Calculating lengths per sample"),
                )
            )
        sampler = PerDatasetSampler.build_sampler_from_config(
            training_conf,
            wh_train.datasets,
            rank=training_conf.local_rank,
            world_size=training_conf.world_size,
            samples_length=samples_length,
            verbose=show_dataset_stats,
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
        sampler = None
        w_sampler = None

    optimizer = OptimizerNames.ADAMW_BNB if training_conf.quantization else OptimizerNames.ADAMW_HF

    if training_conf.quantization:
        import bitsandbytes

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bitsandbytes.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, "weight", {"optim_bits": 32}
                )

    if training_conf.fuse_gelu:
        model = fuse_gelu(model)

    output_dir = (
        training_conf.output_dir
        if training_conf.output_dir
        else f"{training_conf.model_name}-{training_conf.log_dir}-finetuned"
    )

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
    optimizer = AdamW(model.parameters(), lr=float(training_conf.learning_rate), weight_decay=float(training_conf.weight_decay))

    if not training_conf.log_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if training_conf.log_wandb and (not training_conf.deepspeed or training_conf.local_rank == 0):
        wandb.init(
            project="reward-model",
            #entity=training_conf.wandb_entity,
            resume=training_conf.resume_from_checkpoint,
            name=f"{training_conf.model_name}-{training_conf.log_dir}-rm",
            config=training_conf,
        )
    compute_metrics = RewardMetrics(training_conf.metrics)
    trainer = RMTrainer(
        model=model,
        args=args,
        sampler=sampler,
        train_collate_fn=train_collate_fn,
        loss_function=training_conf.loss_fn,
        score_l2_reg=training_conf.score_l2_reg,
        train_dataset=wh_train,
        eval_dataset=wh_evals,
        data_collator=eval_collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_dataloader = trainer.get_train_dataloader()
    w_train_dataloader = trainer.get_w_train_dataloader(w_train, w_train_collate_fn, w_sampler)

    wh_eval_dataloaders = {k : trainer.get_eval_dataloader(wh_eval, eval_collate_fn) for (k, wh_eval) in wh_evals.items()}
    w_eval_dataloaders = {k : trainer.get_eval_dataloader(w_eval, w_eval_collate_fn) for (k, w_eval) in w_evals.items()}

    num_training_steps = training_conf.num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # usually, the train_dataloader will be larger than the w_train_dataloader
    n_itr_per_epoch = len(train_dataloader)
    print(f"{n_itr_per_epoch=}")
    for epoch in range(training_conf.num_train_epochs):
        sampler.set_epoch(epoch)
        w_sampler.set_epoch(epoch)
        for i in tqdm(range(n_itr_per_epoch)):
            default_batch_tuple = next(enumerate(train_dataloader))[1]#[0]
            #print(f"[len batch]: {len(batch)}")

            batch = {k: v.to(device) for k, v in default_batch_tuple[0].items()}
            #outputs = model(**batch)
            batch_tuple = (batch, default_batch_tuple[1])
            loss, outputs = trainer.compute_loss(model, batch_tuple, return_logits=True)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # train with data of [0,...,1,...,0] preference
            default_batch_tuple = next(enumerate(w_train_dataloader))[1]
            # print(f"{len(default_batch_tuple)=}") # 3
            # print(f"{default_batch_tuple[0].keys()=}")

            batch = {k: v.to(device) for k, v in default_batch_tuple[0].items()}
            default_batch_tuple[1].to(device) # move preferences to current device
            batch_tuple = (batch, default_batch_tuple[1], default_batch_tuple[2])
            loss, outputs = trainer.compute_w_loss(model, batch_tuple, return_logits=True)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if i > 0 and i % 1000 == 0:
                print(f"[{epoch=}, EVALUATING W_H DATA]:")
                for dataset_name, wh_eval in wh_eval_dataloaders.items():
                    score_dict = defaultdict(float)
                    # print(f"{type(wh_eval)=}") # dataloader
                    for tmp_id, data in enumerate(wh_eval):
                        #print(data)
                        eval_pred = batch_inference(data, model)
                        results = compute_metrics(eval_pred)
                        for metric in training_conf.metrics:
                            score_dict[metric] += results.get(metric)

                    score_dict = {k: round(v / len(wh_eval), 3) for k, v in score_dict.items()}
                    #print(f"{score_dict}")
                    log_dict = {dataset_name+"_" + k:float(v) for k, v in score_dict.items()}
                    #type_dict = {k: type(v) for k, v in log_dict.items()}
                    #print(f"{score_dict=}, {log_dict=}, {type_dict=}")
                    #for k, v in log_dict.items():
                    #    wandb.log({k: v}, step=i)
                    wandb.log(log_dict, step=epoch * n_itr_per_epoch + i)

                print(f"[{epoch=}, EVALUATING W DATA]:")
                for dataset_name, w_eval in w_eval_dataloaders.items():
                    score_dict = defaultdict(float)

                    for tmp_id, data in enumerate(w_eval):
                        eval_pred = batch_w_inference(data, model)
                        results = compute_metrics(eval_pred)
                        for metric in training_conf.metrics:
                            score_dict[metric] += results.get(metric)

                    score_dict = {k: round(v / len(w_eval), 3) for k, v in score_dict.items()}

                    wandb.log({dataset_name+"_" + k:v for k, v in score_dict.items()}, step=epoch * n_itr_per_epoch + i)
    #trainer.train(resume_from_checkpoint=training_conf.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()