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
from tokenizers import pre_tokenizers

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