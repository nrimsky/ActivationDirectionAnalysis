"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Usage:
python generate_vectors.py --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --use_base_model --model_size 7b --data_path datasets/test.json --dataset_name test
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
import argparse
from typing import List
from utils.tokenize_llama import tokenize_llama
from utils.helpers import make_tensor_save_suffix

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
SAVE_VECTORS_PATH = "vectors"


class ComparisonDataset(Dataset):
    def __init__(self, data_path, system_prompt, token, model_name_path, use_chat):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat

    def prompt_to_tokens(self, instruction, model_output):
        tokens = tokenize_llama(
            self.tokenizer,
            self.system_prompt,
            [(instruction, model_output)],
            no_final_eos=True,
            chat_model=self.use_chat,
        )
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens


def generate_save_vectors(
    layers: List[int], use_base_model: bool, model_size: str, data_path: str, dataset_name: str
):
    """
    layers: list of layers to generate vectors for
    use_base_model: Whether to use the base model instead of the chat model
    model_size: size of the model to use, either "7b" or "13b"
    """
    if not os.path.exists(SAVE_VECTORS_PATH):
        os.makedirs(SAVE_VECTORS_PATH)

    model = LlamaWrapper(
        HUGGINGFACE_TOKEN, SYSTEM_PROMPT, size=model_size, use_chat=not use_base_model
    )
    model.reset_all()

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])

    dataset = ComparisonDataset(
        data_path,
        SYSTEM_PROMPT,
        HUGGINGFACE_TOKEN,
        model.model_name_path,
        model.use_chat,
    )

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, -1, :].detach().cpu()
            pos_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -1, :].detach().cpu()
            neg_activations[layer].append(n_activations)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)
        t.save(
            vec,
            os.path.join(
                SAVE_VECTORS_PATH,
                f"vec_layer_{make_tensor_save_suffix(layer, model.model_name_path, dataset_name)}.pt",
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--data_path", type=str, default="datasets/test.json")
    parser.add_argument("--dataset_name", type=str, default="test")

    args = parser.parse_args()
    generate_save_vectors(
        args.layers, args.use_base_model, args.model_size, args.data_path, args.dataset_name
    )
