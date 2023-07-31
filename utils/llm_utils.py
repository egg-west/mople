import argparse
import copy
import math
import random
from distutils.util import strtobool
from pathlib import Path
from typing import List, NamedTuple

import evaluate
import torch
import transformers
import yaml
from peft import get_peft_model, LoraConfig
from tokenizers import pre_tokenizers

from rewardmodel.models.reward_model import GPTNeoXRewardModel, GPTNeoXMORewardModel, GPTNeoXMORewardModel_W, GPTNeoXMORewardModel_Conservative, GPTNeoXMORewardModelMultiHead
from rewardmodel.models.prefix_llama import LlamaForCausalLM
from rewardmodel.models.patching import patch_model
from rewardmodel.models import freeze_top_n_layers, get_specific_model
from dataset.formatting import QA_SPECIAL_TOKENS

class SpecialTokens(NamedTuple):
    pad_token: str = ""
    eos_token: str = ""
    sep_token: str = ""

class TokenizerConfig(NamedTuple):
    special_tokens: SpecialTokens = {}

TOKENIZER_CONFIGS = {
    "galactica": TokenizerConfig(special_tokens=SpecialTokens("<pad>", "</s>")),
    "GPT-JT": TokenizerConfig(special_tokens=SpecialTokens(sep_token="<|extratoken_100|>")),
    "codegen": TokenizerConfig(special_tokens=SpecialTokens("<|endoftext|>", sep_token="<|endoftext|>")),
    "pythia": TokenizerConfig(special_tokens=SpecialTokens("<|padding|>", "<|endoftext|>", "<|endoftext|>")),
    "gpt-neox": TokenizerConfig(special_tokens=SpecialTokens("<|padding|>", "<|endoftext|>", "<|endoftext|>")),
    "llama": TokenizerConfig(special_tokens=SpecialTokens("</s>", "</s>", sep_token="<s>")),
    "cerebras": TokenizerConfig(special_tokens=SpecialTokens("<|endoftext|>", "<|endoftext|>", "<|endoftext|>")),
    "deberta-v3": TokenizerConfig(special_tokens=SpecialTokens("[PAD]", "[SEP]", sep_token="[CLS]")),
    "bloom": TokenizerConfig(special_tokens=SpecialTokens("<pad>", "</s>", "<s>")),
    "electra": TokenizerConfig(special_tokens=SpecialTokens("[PAD]", "[SEP]", sep_token="[CLS]")),
    "falcon": TokenizerConfig(
        special_tokens=SpecialTokens("<|endoftext|>", "<|endoftext|>", sep_token="<|endoftext|>")
    ),
}

def match_tokenizer_name(model_name: str) -> TokenizerConfig:
    """
    Match a partial model name to a tokenizer configuration
    i.e. model_name `Salesforce/codegen-2B-multi` has config name `codegen`
    """
    tokenizer_config_matches = [config for name, config in TOKENIZER_CONFIGS.items() if name in model_name]
    if not tokenizer_config_matches:
        raise ValueError(f"Cannot find any tokeniser configuration to match {model_name=}")
    elif 1 < len(tokenizer_config_matches):
        raise ValueError(f"Found multiple tokeniser configuration matches for {model_name=}")
    else:
        return tokenizer_config_matches[0]

def get_tokenizer(conf) -> transformers.AutoTokenizer:
    tokenizer_name = conf.model_name

    if "cerebras" in conf.model_name:
        # Only 13B has a tokenizer available on HF
        tokenizer_name = "cerebras/Cerebras-GPT-13B"

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=conf.cache_dir)

    tokenizer_config = match_tokenizer_name(conf.model_name)

    if hasattr(conf, "per_digit_tokens") and conf.per_digit_tokens:
        tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

    if tokenizer_config.special_tokens:
        if "GPT-JT" in conf.model_name:
            tokenizer_config.special_tokens.pad_token = tokenizer.eos_token
        # SpecialTokens : latest in 4.25, 4.26
        tokenizer.add_special_tokens(
            {
                "pad_token": tokenizer_config.special_tokens.pad_token,
                "eos_token": tokenizer_config.special_tokens.eos_token,
                "sep_token": tokenizer_config.special_tokens.sep_token,
            }
        )

    additional_special_tokens = (
        []
        if "additional_special_tokens" not in tokenizer.special_tokens_map
        else tokenizer.special_tokens_map["additional_special_tokens"]
    )
    additional_special_tokens = list(set(additional_special_tokens + list(QA_SPECIAL_TOKENS.values())))

    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    return tokenizer

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def print_parameter_names(model):
    """
    Prints the number of trainable parameters in the model.
    """
    key_list = []
    for key, param in model.named_parameters():
        key_list.append(key)
    print(f"{key_list=}")

def get_peftmodel(conf, tokenizer, pad_vocab_size_to_multiple_of=16, check_freeze_layer=True):
    dtype = torch.float32
    if conf.dtype in ["fp16", "float16"]:
        dtype = torch.float16
    elif conf.dtype in ["bf16", "bfloat16"]:
        dtype = torch.bfloat16

    if conf.is_reward_model:
        if "pythia" in conf.model_name:
            model = GPTNeoXRewardModel.from_pretrained(conf.model_name, cache_dir=conf.cache_dir, torch_dtype=dtype)

            if conf.pooling:
                assert conf.pooling in ("mean", "last"), f"invalid pooling configuration '{conf.pooling}'"
                model.config.pooling = conf.pooling
            #print(f"{model.config=}")
            print_parameter_names(model)
            print("loading LoRA")
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=False,
            )
            model = get_peft_model(model, config)
            print_trainable_parameters(model)

        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                conf.model_name, cache_dir=conf.cache_dir, num_labels=1, torch_dtype=dtype
            )
    if not conf.is_reward_model:
        if conf.peft_type is not None and conf.peft_type == "prefix-tuning" and "llama" in conf.model_name:
            model = LlamaForCausalLM.from_pretrained(conf.model_name, cache_dir=conf.cache_dir, torch_dtype=dtype)
        else:
            model = get_specific_model(
                conf.model_name,
                cache_dir=conf.cache_dir,
                quantization=conf.quantization,
                seq2seqmodel=conf.seq2seqmodel,
                without_head=conf.is_reward_model,
                torch_dtype=dtype,
            )

        n_embs = model.get_input_embeddings().num_embeddings
        if len(tokenizer) != n_embs and check_freeze_layer:
            assert not conf.freeze_layer, "Cannot change the number of embeddings if the model is frozen."

        if len(tokenizer) != n_embs or pad_vocab_size_to_multiple_of:
            p = pad_vocab_size_to_multiple_of
            target_size = len(tokenizer) if not p else math.ceil(len(tokenizer) / p) * p
            print("Resizing embeddings to", target_size)
            model.resize_token_embeddings(target_size)

        if conf.freeze_layer:
            model = freeze_top_n_layers(model, conf.freeze_layer)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print("Number of trainable parameters: {}M".format(int(params / 1e6)))

    patch_model(
        model,
        resid_pdrop=conf.residual_dropout,
        flash_attention=conf.use_flash_attention,
        residual_dropout_lima=conf.residual_dropout_lima,
    )

    return model


def get_model(conf, tokenizer, pad_vocab_size_to_multiple_of=16, check_freeze_layer=True):
    dtype = torch.float32
    if conf.dtype in ["fp16", "float16"]:
        dtype = torch.float16
    elif conf.dtype in ["bf16", "bfloat16"]:
        dtype = torch.bfloat16

    if conf.is_reward_model:
        if "pythia" in conf.model_name:
            model = GPTNeoXRewardModel.from_pretrained(conf.model_name, cache_dir=conf.cache_dir, torch_dtype=dtype)

            if conf.pooling:
                assert conf.pooling in ("mean", "last"), f"invalid pooling configuration '{conf.pooling}'"
                model.config.pooling = conf.pooling
        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                conf.model_name, cache_dir=conf.cache_dir, num_labels=1, torch_dtype=dtype
            )
    if not conf.is_reward_model:
        if conf.peft_type is not None and conf.peft_type == "prefix-tuning" and "llama" in conf.model_name:
            model = LlamaForCausalLM.from_pretrained(conf.model_name, cache_dir=conf.cache_dir, torch_dtype=dtype)
        else:
            model = get_specific_model(
                conf.model_name,
                cache_dir=conf.cache_dir,
                quantization=conf.quantization,
                seq2seqmodel=conf.seq2seqmodel,
                without_head=conf.is_reward_model,
                torch_dtype=dtype,
            )

        n_embs = model.get_input_embeddings().num_embeddings
        if len(tokenizer) != n_embs and check_freeze_layer:
            assert not conf.freeze_layer, "Cannot change the number of embeddings if the model is frozen."

        if len(tokenizer) != n_embs or pad_vocab_size_to_multiple_of:
            p = pad_vocab_size_to_multiple_of
            target_size = len(tokenizer) if not p else math.ceil(len(tokenizer) / p) * p
            print("Resizing embeddings to", target_size)
            model.resize_token_embeddings(target_size)

        if conf.freeze_layer:
            model = freeze_top_n_layers(model, conf.freeze_layer)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print("Number of trainable parameters: {}M".format(int(params / 1e6)))

    patch_model(
        model,
        resid_pdrop=conf.residual_dropout,
        flash_attention=conf.use_flash_attention,
        residual_dropout_lima=conf.residual_dropout_lima,
    )

    return model

def get_momodel(conf, tokenizer, pad_vocab_size_to_multiple_of=16, check_freeze_layer=True):
    dtype = torch.float32
    if conf.dtype in ["fp16", "float16"]:
        dtype = torch.float16
    elif conf.dtype in ["bf16", "bfloat16"]:
        dtype = torch.bfloat16

    assert conf.is_reward_model == True
    if "pythia" in conf.model_name:
        model = GPTNeoXMORewardModel.from_pretrained(conf.model_name, cache_dir=conf.cache_dir, torch_dtype=dtype)

        if conf.pooling:
            assert conf.pooling in ("mean", "last"), f"invalid pooling configuration '{conf.pooling}'"
            model.config.pooling = conf.pooling
    else:
        raise NotImplementedError
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            conf.model_name, cache_dir=conf.cache_dir, num_labels=1, torch_dtype=dtype
        )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print("Number of trainable parameters: {}M".format(int(params / 1e6)))

    patch_model(
        model,
        resid_pdrop=conf.residual_dropout,
        flash_attention=conf.use_flash_attention,
        residual_dropout_lima=conf.residual_dropout_lima,
    )

    return model


def get_momodel_w(conf, tokenizer, pad_vocab_size_to_multiple_of=16, check_freeze_layer=True):
    dtype = torch.float32
    if conf.dtype in ["fp16", "float16"]:
        dtype = torch.float16
    elif conf.dtype in ["bf16", "bfloat16"]:
        dtype = torch.bfloat16

    assert conf.is_reward_model == True
    if "pythia" in conf.model_name:
        model = GPTNeoXMORewardModel_W.from_pretrained(conf.model_name, cache_dir=conf.cache_dir, torch_dtype=dtype)

        if conf.pooling:
            assert conf.pooling in ("mean", "last"), f"invalid pooling configuration '{conf.pooling}'"
            model.config.pooling = conf.pooling
    else:
        raise NotImplementedError
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            conf.model_name, cache_dir=conf.cache_dir, num_labels=1, torch_dtype=dtype
        )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print("Number of trainable parameters: {}M".format(int(params / 1e6)))

    patch_model(
        model,
        resid_pdrop=conf.residual_dropout,
        flash_attention=conf.use_flash_attention,
        residual_dropout_lima=conf.residual_dropout_lima,
    )

    return model

def get_momodel_multi_head(conf, tokenizer, pad_vocab_size_to_multiple_of=16, check_freeze_layer=True):
    dtype = torch.float32
    if conf.dtype in ["fp16", "float16"]:
        dtype = torch.float16
    elif conf.dtype in ["bf16", "bfloat16"]:
        dtype = torch.bfloat16

    assert conf.is_reward_model == True
    if "pythia" in conf.model_name:
        model = GPTNeoXMORewardModelMultiHead.from_pretrained(conf.model_name, cache_dir=conf.cache_dir, torch_dtype=dtype)

        if conf.pooling:
            assert conf.pooling in ("mean", "last"), f"invalid pooling configuration '{conf.pooling}'"
            model.config.pooling = conf.pooling
    else:
        raise NotImplementedError
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            conf.model_name, cache_dir=conf.cache_dir, num_labels=1, torch_dtype=dtype
        )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print("Number of trainable parameters: {}M".format(int(params / 1e6)))

    patch_model(
        model,
        resid_pdrop=conf.residual_dropout,
        flash_attention=conf.use_flash_attention,
        residual_dropout_lima=conf.residual_dropout_lima,
    )

    return model

def get_momodel_conservative(conf, tokenizer, pad_vocab_size_to_multiple_of=16, check_freeze_layer=True):
    dtype = torch.float32
    if conf.dtype in ["fp16", "float16"]:
        dtype = torch.float16
    elif conf.dtype in ["bf16", "bfloat16"]:
        dtype = torch.bfloat16

    assert conf.is_reward_model == True
    if "pythia" in conf.model_name:
        model = GPTNeoXMORewardModel_Conservative.from_pretrained(conf.model_name, cache_dir=conf.cache_dir, torch_dtype=dtype)

        if conf.pooling:
            assert conf.pooling in ("mean", "last"), f"invalid pooling configuration '{conf.pooling}'"
            model.config.pooling = conf.pooling
    else:
        raise NotImplementedError
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            conf.model_name, cache_dir=conf.cache_dir, num_labels=1, torch_dtype=dtype
        )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print("Number of trainable parameters: {}M".format(int(params / 1e6)))

    patch_model(
        model,
        resid_pdrop=conf.residual_dropout,
        flash_attention=conf.use_flash_attention,
        residual_dropout_lima=conf.residual_dropout_lima,
    )

    return model