from pathlib import Path
import json
from tokenizers import Tokenizer

class LFM2Tokenizer:
    def __init__(self, tokenizer_file_path: str, tokenizer_config_path: str = None, model_config=None):
        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))

        tokenizer_config = {}
        if tokenizer_config_path is not None:
            with Path(tokenizer_config_path).open(encoding="utf-8") as config_file:
                tokenizer_config = json.load(config_file)

        self.bos_token = tokenizer_config.get("bos_token", "<|startoftext|>")
        self.eos_token = tokenizer_config.get("eos_token", "<|im_end|>")
        self.pad_token = tokenizer_config.get("pad_token", "<|pad|>")

        self.bos_id = self._tok.token_to_id(self.bos_token)
        self.eos_id = self._tok.token_to_id(self.eos_token)
        self.pad_id = self._tok.token_to_id(self.pad_token)

        if model_config is not None:
            expected_ids = {
                "bos": model_config.bos_token_id,
                "eos": model_config.eos_token_id,
                "pad": model_config.pad_token_id,
            }
            actual_ids = {"bos": self.bos_id, "eos": self.eos_id, "pad": self.pad_id}
            if actual_ids != expected_ids:
                raise ValueError(f"Tokenizer special-token IDs do not match model config: {actual_ids} != {expected_ids}")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def apply_chat_template(self, prompt: str, system_prompt: str = None) -> str:
        """
        Formats a single-turn prompt according to the LFM2.5 chat template.
        """
        system_message = ""
        if system_prompt:
            system_message = f"<|im_start|>system\n{system_prompt}{self.eos_token}\n"

        return (
            f"{self.bos_token}{system_message}"
            f"<|im_start|>user\n{prompt}{self.eos_token}\n"
            f"<|im_start|>assistant\n<think>"
        )
