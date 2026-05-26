"""Python wrapper for KVCacheConfig with additional convenience methods."""

import json
from typing import Any, Optional

from rtp_llm.ops import KVCacheConfig as CppKVCacheConfig


class KVCacheConfig(CppKVCacheConfig):
    """Python wrapper for C++ KVCacheConfig with additional convenience methods."""

    def load_and_update_task_prompt_config(
        self, tokenizer: Optional[Any] = None
    ) -> None:
        """Load task prompt configuration and update token IDs if tokenizer is provided.

        This method combines the functionality of load_task_prompt_config and
        update_task_prompt_tokens_id. It loads the configuration from either
        multi_task_prompt file or multi_task_prompt_str, stores it in
        multi_task_prompt_config, and if a tokenizer is provided, updates the
        token IDs for each task prompt.

        Args:
            tokenizer: Optional tokenizer instance with encode method. If provided,
                     will update task prompt tokens from the loaded configuration.
        """
        # Load task prompt configuration
        if self.multi_task_prompt:
            with open(self.multi_task_prompt, "r") as reader:
                multi_task_prompt_config = json.loads(reader.read(), strict=False)
        elif self.multi_task_prompt_str != "":
            multi_task_prompt_config = json.loads(
                self.multi_task_prompt_str, strict=False
            )
        else:
            return

        # Update task prompt tokens if tokenizer is provided
        if tokenizer and multi_task_prompt_config:
            # The default tokenizer.encode(prompt) uses add_special_tokens=True and
            # appends a trailing special token (e.g. <|endoftext|> for Qwen).
            # Request user-text never carries that token, so leaving it in would
            # break exact-prefix matching for the embedding resident prefix cache
            # (and inject a spurious special token between system prompt and user
            # input on NormalEngine's task_id prepend path). Encode with
            # add_special_tokens=False to record exactly the prompt's user-text tokens.
            if isinstance(multi_task_prompt_config, list):
                for info in multi_task_prompt_config:
                    task_id: str = str(info["task_id"])
                    prompt: str = info["prompt"]
                    tokens_id = tokenizer.encode(prompt, add_special_tokens=False)
                    self.insertMultiTaskPromptTokens(task_id, tokens_id)

        if self.multi_task_prompt_tokens:
            self.reuse_cache = True
