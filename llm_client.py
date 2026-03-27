"""LLM client wrapper with caching, retries, and token tracking."""

import hashlib
import json
import logging
import time
from typing import Any

from openai import OpenAI

import config

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around OpenAI-compatible API with caching and retries."""

    def __init__(self):
        self.client = OpenAI(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
        )
        self.model = config.OPENAI_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.max_retries = config.LLM_MAX_RETRIES

        # In-memory cache: hash(messages) -> response content
        self._cache: dict[str, str] = {}

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def _cache_key(self, messages: list[dict]) -> str:
        raw = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 16000,
        use_cache: bool = True,
    ) -> str:
        """
        Send a chat completion request.
        
        Returns:
            The assistant's response content as a string.
        """
        temp = temperature if temperature is not None else self.temperature

        # Check cache
        if use_cache:
            key = self._cache_key(messages)
            if key in self._cache:
                logger.debug("Cache hit for prompt")
                return self._cache[key]

        # Retry loop
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                )

                # Track tokens
                if response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                self.total_calls += 1

                content = response.choices[0].message.content or ""

                # Cache result
                if use_cache:
                    self._cache[key] = content

                return content

            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"LLM call attempt {attempt}/{self.max_retries} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} attempts: {last_error}"
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 16000,
    ) -> Any:
        """Send a chat request and parse the response as JSON."""
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        # Try to extract JSON from markdown code blocks if present
        raw = raw.strip()
        if raw.startswith("```"):
            # Remove ```json ... ``` wrapper
            lines = raw.split("\n")
            # Find start and end of code block
            start = 1 if lines[0].startswith("```") else 0
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            raw = "\n".join(lines[start:end])

        return json.loads(raw)

    def get_usage_summary(self) -> str:
        """Return a summary of token usage."""
        total = self.total_input_tokens + self.total_output_tokens
        return (
            f"LLM Usage: {self.total_calls} calls, "
            f"{self.total_input_tokens:,} input tokens, "
            f"{self.total_output_tokens:,} output tokens, "
            f"{total:,} total tokens"
        )
