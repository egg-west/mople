import copy
import yaml
from typing import List, NamedTuple, Optional
from pathlib import Path

from torch.utils.data import ConcatDataset, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from dataset.oasst_dataset import load_oasst_export
from dataset.hh_dataset import load_anthropic_rlhf_helpful, load_anthropic_rlhf_harmless

RL_DATASETS = [
    "oasst_export",
    "webgpt",
    "private_tuning",
    "alpaca",
    "hf_summary",
    "hf_summary_pairs",
    "vicuna",
]

RM_DATASETS = [
    "oasst_export",
    "augment_oasst",
    "anthropic_rlhf",
    "anthropic_rlhf_helpful",
    "anthropic_rlhf_harmless",
    "hf_summary",
    "hf_summary_pairs",
    "shp",
    "hellaswag",
    "webgpt",
]

def get_one_dataset(
    conf,
    dataset_name: str,
    val_split: float = 0.2,
    data_path: str = None,
    mode: str = "sft",
    max_val_set: Optional[int] = None,
    **kwargs,
) -> tuple[Dataset, Dataset | None]:
    if mode == "rl":
        assert dataset_name in RL_DATASETS, f"Dataset {dataset_name} not supported for RL"

    if mode == "rm":
        assert dataset_name in RM_DATASETS, f"Dataset {dataset_name} not supported for reward modeling"

    data_path = data_path or conf.cache_dir
    dataset_name = dataset_name.lower()

    if dataset_name == "oasst_export":
        train, eval = load_oasst_export(data_path=data_path, val_split=val_split, mode=mode, **kwargs)
    elif dataset_name == "anthropic_rlhf_helpful":
        train, eval = load_anthropic_rlhf_helpful()
    elif dataset_name == "anthropic_rlhf_harmless":
        train, eval = load_anthropic_rlhf_harmless()
    else:
        raise NotImplementedError("Dataset but oasst_export not implemented")
    """
    if dataset_name in QA_DATASETS:
        dataset = QADataset(dataset_name, data_path, "train")
        if not dataset.no_val:
            eval = QADataset(dataset_name, data_path, "validation")
            train = dataset
    elif dataset_name in SUMMARIZATION_DATASETS:
        dataset = SummarizationDataset(dataset_name, data_path, "train")
        if dataset_name != "debate_sum":
            eval = SummarizationDataset(dataset_name, data_path, "validation")
            train = dataset
    elif dataset_name in INSTRUCTION_DATASETS:
        dataset = InstructionDataset(dataset_name, data_path, "train")
    elif "ted_trans" in dataset_name:
        language_pair = dataset_name.split("_")[-1]
        dataset = TEDTalk(pair=language_pair, split="train")
    elif "wmt2019" in dataset_name:
        language_pair = dataset_name.split("_")[-1]
        train = WMT2019(pair=language_pair, split="train")
        eval = WMT2019(pair=language_pair, split="validation")
    elif dataset_name == "dive_mt":
        dataset = DiveMT()
    elif dataset_name == "webgpt":
        dataset = WebGPT(mode=mode)
    elif dataset_name in ("alpaca", "code_alpaca"):
        train, eval = load_alpaca_dataset(dataset_name, val_split=val_split, cache_dir=data_path, **kwargs)
    elif dataset_name == "gpt4all":
        dataset = Gpt4All(mode=mode, cache_dir=data_path)
    elif dataset_name == "prosocial_dialogue":
        train = ProsocialDialogue(cache_dir=data_path, split="train")
        eval = ProsocialDialogue(cache_dir=data_path, split="validation")
    elif dataset_name == "explain_prosocial":
        train = ProsocialDialogueExplaination(cache_dir=data_path, split="train")
        eval = ProsocialDialogueExplaination(cache_dir=data_path, split="validation")
    elif dataset_name == "soda":
        dataset = SODA(data_path, **kwargs)
    elif dataset_name == "soda_dialogue":
        dataset = SODADialogue(data_path)
    elif dataset_name == "joke":
        dataset = JokeExplaination(data_path)
    elif dataset_name == "oa_translated":
        # TODO make val_split lower..? by saganos
        dataset = TranslatedQA(data_path)
    elif dataset_name == "vicuna":
        dataset = Vicuna(cache_dir=data_path, **kwargs)
    elif dataset_name == "oasst_export":
        train, eval = load_oasst_export(data_path=data_path, val_split=val_split, mode=mode, **kwargs)
    elif dataset_name == "hf_summary":
        train = HFSummary(split="train", mode=mode)
        eval = HFSummary(split="valid1", mode=mode)
    elif dataset_name == "hf_summary_pairs":
        train = HFSummaryPairs(split="train", mode=mode)
        eval = HFSummaryPairs(split="valid1", mode=mode)
    elif dataset_name == "augment_oasst":
        # reward model mode only
        assert mode == "rm"
        train = AugmentedOA(data_path + "/" + kwargs["input_file_path"], split="train")
        eval = AugmentedOA(data_path + "/" + kwargs["input_file_path"], split="val")
    elif dataset_name == "oig_file":
        train, eval = load_oig_file(val_split=val_split, **kwargs)
    elif dataset_name == "anthropic_rlhf":
        train, eval = load_anthropic_rlhf()
    elif dataset_name == "shp":
        train, eval = load_shp()
    elif dataset_name == "hellaswag":
        train, eval = load_hellaswag()
    elif dataset_name == "dolly15k":
        dataset = DatabricksDolly15k(cache_dir=data_path, mode=mode, **kwargs)
    elif dataset_name == "alpaca_gpt4":
        dataset = AlpacaGpt4(cache_dir=data_path, mode=mode, **kwargs)
    elif dataset_name == "red_pajama":
        dataset = RedPajama(cache_dir=data_path, mode=mode, **kwargs)
    elif dataset_name == "gpteacher_roleplay":
        dataset = GPTeacher_Roleplay(cache_dir=data_path, mode=mode, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # if eval not already defined
    if not ("eval" in locals() and "train" in locals()):
        train, eval = train_val_dataset(dataset, val_split=val_split)

    if eval and max_val_set and len(eval) > max_val_set:
        subset_indices = np.random.choice(len(eval), max_val_set)
        eval = Subset(eval, subset_indices)
    """
    return train, eval

def get_dataset_name_and_kwargs_from_data_config(data_config):
    if isinstance(data_config, dict):
        name = list(data_config.keys())[0]

        # first copy the dict, then remove the size and fraction
        kwargs = copy.deepcopy(data_config[name])

        kwargs.pop("fraction", None)
        kwargs.pop("size", None)
        return name, kwargs
    else:
        return data_config, {}

def get_dataset(
    conf,
    mode: str = "sft",
) -> tuple[ConcatDataset, dict[str, Subset]]:
    train_datasets, evals = [], {}

    for data_config in conf.datasets + conf.datasets_extra:
        dataset_name, kwargs = get_dataset_name_and_kwargs_from_data_config(data_config)
        train, val = get_one_dataset(conf, dataset_name, mode=mode, **kwargs)
        train_datasets.append(train)

        if val is not None:
            evals[dataset_name] = Subset(val, list(range(min(len(val), conf.eval_size)))) if conf.eval_size else val

    train = ConcatDataset(train_datasets)

    return train, evals

def read_yamls(dir):
    conf = {}
    no_conf = True

    for config_file in Path(dir).glob("**/*.yaml"):
        no_conf = False
        with config_file.open("r") as f:
            conf.update(yaml.safe_load(f))

    if no_conf:
        print(f"WARNING: No yaml files found in {dir}")

    return conf

def get_modataset(
    conf,
    mode: str = "sft",
) -> tuple[ConcatDataset, dict[str, Subset]]:
    """generate multi-objective datasets

    Args:
        conf (yaml): configurations for dataset loading
        mode (str, optional): types of training. Defaults to "sft".

    Returns:
        tuple[ConcatDataset, dict[str, Subset]]: w_h train without preference, w_train with preference set as [0,...1,...0]
    """
    wh_train_datasets, w_train_datasets, evals = [], [], {}

    for data_config in conf.datasets + conf.datasets_extra:
        dataset_name, kwargs = get_dataset_name_and_kwargs_from_data_config(data_config)
        train, val = get_one_dataset(conf, dataset_name, mode=mode, **kwargs)
        wh_train_datasets.append(train)

        if val is not None:
            evals[dataset_name] = Subset(val, list(range(min(len(val), conf.eval_size)))) if conf.eval_size else val

    for data_config in conf.w_datasets:
        dataset_name, kwargs = get_dataset_name_and_kwargs_from_data_config(data_config)
        train, val = get_one_dataset(conf, dataset_name, mode=mode, **kwargs)
        w_train_datasets.append(train)

    wh_train = ConcatDataset(wh_train_datasets)
    w_train = ConcatDataset(w_train_datasets)

    return wh_train, w_train, evals