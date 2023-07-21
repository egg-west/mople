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
