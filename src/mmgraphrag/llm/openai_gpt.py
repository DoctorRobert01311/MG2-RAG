import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from openai import AzureOpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        gen_params = (
            getattr(self, "llm_config", {}).generate_params
            if hasattr(self, "llm_config")
            else {}
        )
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))
        key_data = {
            "messages": messages,
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        lock_file = self.cache_file_name + ".lock"

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """
            )
            conn.commit()
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                logger.info(f"[llm] Cache HIT for key {key_hash[:8]}")
                return message, metadata, True

        result = func(self, *args, **kwargs)
        message, metadata, _ = result

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """
            )
            metadata_str = json.dumps(metadata)
            c.execute(
                "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                (key_hash, message, metadata_str),
            )
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper


def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)

    return wrapper


class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""

    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        config_dict["max_retries"] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(
        self,
        cache_dir,
        global_config,
        cache_filename: str = None,
        high_throughput: bool = True,
        **kwargs,
    ) -> None:

        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(
                limits=limits, timeout=httpx.Timeout(5 * 60, read=5 * 60), trust_env=False
            )
        else:
            client = None

        self.max_retries = kwargs.get("max_retries", 2)

        self.openai_client = OpenAI(
            base_url=self.llm_base_url,
            api_key=os.getenv("OPENAI_API_KEY") or "sk-local",
            http_client=client,
            max_retries=self.max_retries,
        )
        logger.info(f"[llm] CacheOpenAI initialized. Model: {self.llm_name}, Cache: {self.cache_file_name}")

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict["llm_name"] = self.global_config.llm_name
        config_dict["llm_base_url"] = self.global_config.llm_base_url
        config_dict["generate_params"] = {
            "model": self.global_config.llm_name,
            "max_tokens": config_dict.get("max_new_tokens", 400),
            "n": config_dict.get("num_gen_choices", 1),
            "seed": config_dict.get("seed", 0),
            "temperature": config_dict.get("temperature", 0.0),
            "top_p": config_dict.get("top_p", 1.0),
            "extra_body": {
                "top_k": config_dict.get("sampling_top_k", 20),
                "chat_template_kwargs": {
                    "enable_thinking": config_dict.get("enable_thinking", False)
                },
            },
        }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @dynamic_retry_decorator
    def infer(
        self, messages: List[TextChatMessage], **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")


        response = self.openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        logger.debug(f"input messages: {messages}")
        logger.debug(f"Received response from OpenAI GPT API: {response_message}")
        assert isinstance(response_message, str), "response_message should be a string"

        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }
        logger.debug(f"LLM inference metadata: {metadata}")
        return response_message, metadata, False
