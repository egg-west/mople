import argparse
from distutils.util import strtobool

from utils.llm_utils import get_tokenizer
from utils.dataset_utils import get_dataset, read_yamls

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

    """
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
    """

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
    #tokenizer = get_tokenizer(training_conf)
    #model = get_model(training_conf, tokenizer)
    # test the data loader
    training_conf = argument_parsing()
    train, evals = get_dataset(training_conf, mode="rm")