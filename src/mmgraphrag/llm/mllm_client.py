import hashlib
import json
import os
import sqlite3
import functools
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional

from openai import OpenAI
import openai
import httpx
from packaging import version
from filelock import FileLock

from .base import BaseLLM, LLMConfig
from ..utils.logging_utils import get_logger 

logger = get_logger(__name__)

def cache_mllm_response(func):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        gen_params = getattr(self, "mllm_config", {}).generate_params if hasattr(self, "mllm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))
        key_data = {"messages": messages, "model": model, "temperature": temperature}
        
        try:
            key_str = json.dumps(key_data, sort_keys=True, default=str)
        except Exception as e:
            logger.warning(f"Failed to serialize key for caching: {e}")
            return func(self, *args, **kwargs)

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
                logger.info(f"[mllm] Cache HIT for key {key_hash[:8]}")
                return message, metadata, True
        result = func(self, *args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 3:
            return result
        response_content, metadata, _ = result
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
                (key_hash, response_content, metadata_str),
            )
            conn.commit()
            conn.close()
        return response_content, metadata, False
    return wrapper


class CacheMLLM(BaseLLM):
    def __init__(self,global_config) -> None:
        super().__init__(global_config)
        self.mllm_name = global_config.mllm_name
        self.mllm_base_url = global_config.mllm_base_url
        
        save_dir = global_config.save_dir
        cache_dir = os.path.join(save_dir, "mllm_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        safe_name = self.mllm_name.replace('/', '_').replace(':', '_')
        cache_filename = f"{safe_name}_cache.sqlite"  
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)
        self._init_llm_config()
        self.client = OpenAI(
            base_url=self.mllm_base_url,
            api_key=os.getenv("OPENAI_API_KEY") or "sk-local",
            timeout=600.0,
            http_client=httpx.Client(timeout=600.0, trust_env=False),
        )
        
        logger.info(f"[mllm] CacheMLLM initialized. Model: {self.mllm_name}, Cache: {self.cache_file_name}")

    def _init_llm_config(self) -> None:
        """
        根据 global_config 初始化 self.llm_config
        """
        if hasattr(self.global_config, "to_dict"):
            config_dict = self.global_config.to_dict()
        else:
            config_dict = self.global_config.__dict__
        generate_params = {
            "model": self.global_config.mllm_name,
            "max_tokens": config_dict.get("max_new_tokens", 400),
            "n": config_dict.get("num_gen_choices", 1),
            "seed": config_dict.get("seed", 0),
            "temperature": config_dict.get("temperature", 0.0),
            "top_p": config_dict.get("top_p", 1.0),
            "extra_body":{
                "top_k": config_dict.get("sampling_top_k", 20),
                "chat_template_kwargs": {"enable_thinking": config_dict.get("enable_thinking", False)},
            }, 
        }
        config_dict["generate_params"] = generate_params
        self.mllm_config = LLMConfig.from_dict(config_dict=config_dict)

    @cache_mllm_response
    def infer(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        params = deepcopy(self.mllm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages     

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            logger.error(f"MLLM Inference Failed: {e}")
            raise e
        response_content = response.choices[0].message.content
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
        }
        return response_content, metadata, False
