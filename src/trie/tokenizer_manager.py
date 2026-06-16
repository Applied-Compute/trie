import os

import numpy as np
import structlog

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer

logger = structlog.get_logger(__name__)


class TokenizerManager:
    def __init__(self, model_name: str, seed: int | None = None) -> None:
        self._model_name = model_name
        self._rng = np.random.default_rng(seed)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    def _decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer(text, add_special_tokens=False).input_ids

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def _sample_token_ids(self, prompt_length: int) -> list[int]:
        return self._rng.integers(
            self._tokenizer.vocab_size, size=prompt_length
        ).tolist()

    def get_prompt(
        self,
        prompt_length: int,
        *,
        max_rounds: int = 10,
    ) -> str:
        """Return text that round-trips to ``prompt_length`` tokens, repairing
        decode/encode mismatches for up to ``max_rounds`` iterations."""
        if prompt_length == 0:
            return ""

        token_ids = self._sample_token_ids(prompt_length)
        prompt = self._decode(token_ids)
        retokenized_ids = self._encode(prompt)

        rounds = 0
        while len(retokenized_ids) != prompt_length and rounds < max_rounds:
            if len(retokenized_ids) < prompt_length:
                num_extras = prompt_length - len(retokenized_ids)
                token_ids = retokenized_ids + self._sample_token_ids(num_extras)
            else:
                token_ids = retokenized_ids[:prompt_length]
            prompt = self._decode(token_ids)
            retokenized_ids = self._encode(prompt)
            rounds += 1

        if len(retokenized_ids) != prompt_length:
            logger.warning(
                "Failed to synthesize prompt for "
                f"{self._model_name!r}: target_length={prompt_length}, "
                f"final_length={len(retokenized_ids)}, repair_rounds={rounds}, "
                f"max_rounds={max_rounds}. Returning best effort result."
            )
            return prompt

        return prompt
