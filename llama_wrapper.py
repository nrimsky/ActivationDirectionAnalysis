import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None


class LlamaWrapper:
    def __init__(
        self,
        token,
        system_prompt,
        size="7b",
        use_chat=True
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        self.use_chat = use_chat
        if use_chat:
            self.model_name_path = f"meta-llama/Llama-2-{size}-chat-hf"
        else:
            self.model_name_path = f"meta-llama/Llama-2-{size}-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_path, use_auth_token=token
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name_path, use_auth_token=token
            )
            .half()
            .to(self.device)
        )
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer)

    def get_logits(self, tokens):
        with t.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()