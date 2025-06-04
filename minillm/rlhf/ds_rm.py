"""
Dataset for RLHF
"""

"""
Dataset example:

{
  "chosen": [
    {
      "content": "What beats purity?",
      "role": "user"
    },
    {
      "content": "Practicality.",
      "role": "assistant"
    }
  ],
  "rejected": [
    {
      "content": "What beats purity?",
      "role": "user"
    },
    {
      "content": "Nothing.",
      "role": "assistant"
    }
  ]
}
"""
import json
import os

from torch.utils.data import Dataset


class RMDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length=4096):
        self.data_path = data_path
        self.data: list[dict] = self._load_data()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_data(self):
        samples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)

        # print(f"chosen_prompt: {chosen_prompt}")
        # print(f"rejected_prompt: {rejected_prompt}")

        chosen_input_embedding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_input_embedding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids_chosen": chosen_input_embedding.input_ids.squeeze(0),
            "attention_mask_chosen": chosen_input_embedding.attention_mask.squeeze(0),
            "input_ids_rejected": rejected_input_embedding.input_ids.squeeze(0),
            "attention_mask_rejected": rejected_input_embedding.attention_mask.squeeze(0),
        }


from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ds = RMDataset(
    data_path="data/reward_model.jsonl",
    tokenizer=AutoTokenizer.from_pretrained("minillm/tokenizer"),
    max_length=48,
)

for i in range(len(ds)):
    sample = ds[i]
    print(sample)
    break
