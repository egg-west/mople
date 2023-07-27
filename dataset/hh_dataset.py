import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

class AnthropicRLHF(Dataset):
    name = "anthropic_rlhf"

    @staticmethod
    def _split_dialogue(text: str) -> list[tuple[str, str]]:
        lines = text.split("\n\n")

        dialogue: list[tuple[str, str]] = []

        # go over messages and combine consecutive messages from the
        # same speaker (OA v1 expects alternating roles)
        role = None
        messages = []
        for line in lines:
            if line.startswith("Human:"):
                speaker = "Human"
                message = line[7:]
            elif line.startswith("Assistant:"):
                speaker = "Assistant"
                message = line[11:]
            else:
                continue
            if role != speaker:
                if role is not None:
                    dialogue.append((role, "\n".join(messages)))
                    messages = []
                role = speaker
            messages.append(message.strip())

        if role is not None and len(messages) > 0:
            dialogue.append((role, "\n".join(messages)))

        return dialogue

    def __init__(self, split: str = "train") -> None:
        super().__init__()
        assert split in ("train", "test")
        self.split = split
        self.data = []
        dataset = load_dataset("Anthropic/hh-rlhf")[split]

        for entry in dataset:
            chosen = entry["chosen"]

            if "Assistant" not in chosen:
                continue

            rejected = entry["rejected"]
            chosen = self._split_dialogue(chosen)
            rejected = self._split_dialogue(rejected)
            assert rejected[0][0] == "Human" and chosen[0][0] == "Human"

            # only very few items have non matching lengths
            if len(rejected) == len(chosen):
                prefix = [line for (speaker, line) in chosen[:-1]]
                good_reply = chosen[-1][1]  # last part of dialog, the text
                bad_reply = rejected[-1][1]  # last part of dialog, the text
                self.data.append((prefix, [good_reply, bad_reply]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, list[str]]:
        return self.data[index]

class AnthropicRLHFH(Dataset):
    """load Anthropic/rlhf helpful OR harmless data (`H` in the end)
    """
    name = "anthropic_rlhf"

    @staticmethod
    def _split_dialogue(text: str) -> list[tuple[str, str]]:
        lines = text.split("\n\n")

        dialogue: list[tuple[str, str]] = []

        # go over messages and combine consecutive messages from the
        # same speaker (OA v1 expects alternating roles)
        role = None
        messages = []
        for line in lines:
            if line.startswith("Human:"):
                speaker = "Human"
                message = line[7:]
            elif line.startswith("Assistant:"):
                speaker = "Assistant"
                message = line[11:]
            else:
                continue
            if role != speaker:
                if role is not None:
                    dialogue.append((role, "\n".join(messages)))
                    messages = []
                role = speaker
            messages.append(message.strip())

        if role is not None and len(messages) > 0:
            dialogue.append((role, "\n".join(messages)))

        return dialogue

    def __init__(
        self,
        split: str = "train",
        objective: str = 'helful',
        n_obj: int = None,
        obj_id: int = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "test")
        self.split = split
        self.data = []
        assert objective in ['helpful-base', 'helpful-online', 'helpful-rejection-sampled', 'harmless-base', 'red-team-attempts']
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir=objective)[split]

        for entry in dataset:
            chosen = entry["chosen"]

            if "Assistant" not in chosen:
                continue

            rejected = entry["rejected"]
            chosen = self._split_dialogue(chosen)
            rejected = self._split_dialogue(rejected)
            assert rejected[0][0] == "Human" and chosen[0][0] == "Human"

            # only very few items have non matching lengths
            if len(rejected) == len(chosen):
                prefix = [line for (speaker, line) in chosen[:-1]]
                good_reply = chosen[-1][1]  # last part of dialog, the text
                bad_reply = rejected[-1][1]  # last part of dialog, the text
                if n_obj == None:
                    self.data.append((prefix, [good_reply, bad_reply]))
                else:
                    preference = np.zeros((2, n_obj))
                    preference[0, obj_id] = 1.0
                    preference[1, obj_id] = 1.0
                    self.data.append((prefix, [good_reply, bad_reply], preference))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, list[str]]:
        return self.data[index]

def load_anthropic_rlhf() -> tuple[Dataset, Dataset]:
    train = AnthropicRLHF(split="train")
    validation = AnthropicRLHF(split="test")
    return train, validation

def load_anthropic_rlhf_helpful(n_obj=None, obj_id=None) -> tuple[Dataset, Dataset]:
    train = AnthropicRLHFH(split="train", objective='helpful-base', n_obj=n_obj, obj_id=obj_id)
    validation = AnthropicRLHFH(split="test", objective='helpful-base', n_obj=n_obj, obj_id=obj_id)
    print(f"Anthropic rlhf-helpful dataset: {len(train)=}, {len(validation)=}")
    return train, validation

def load_anthropic_rlhf_harmless(n_obj=None, obj_id=None) -> tuple[Dataset, Dataset]:
    train = AnthropicRLHFH(split="train", objective='harmless-base', n_obj=n_obj, obj_id=obj_id)
    validation = AnthropicRLHFH(split="test", objective='harmless-base', n_obj=n_obj, obj_id=obj_id)
    print(f"Anthropic rlhf-harmless dataset: {len(train)=}, {len(validation)=}")
    return train, validation