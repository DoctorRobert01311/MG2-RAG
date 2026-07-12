import os
from dataclasses import dataclass, fields
from typing import Literal, Union, Optional, List, Dict, get_args, get_origin
import json

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""

    llm_name: str = "gpt-4o-mini"
    llm_base_url: Optional[str] = None
    embedding_base_url: Optional[str] = None
    tokenizer_path: str = "assets/Qwen2Tokenizer"
    azure_endpoint: Optional[str] = None
    azure_embedding_endpoint: Optional[str] = None
    
    max_new_tokens: Optional[int] = 400
    num_gen_choices: int = 1
    seed: Optional[int] = None
    temperature: float = 0
    top_p: float = 1.0
    sampling_top_k: int = 20
    enable_thinking: bool = False
    
    response_format: Optional[dict] = None 

    max_retry_attempts: int = 5
    
    force_openie_from_scratch: bool = False
    force_index_from_scratch: bool = False
    rerank_dspy_file_path: Optional[str] = None

    text_preprocessor_class_name: str = "TextPreprocessor"
    preprocess_encoder_name: str = "gpt-4o"
    preprocess_chunk_overlap_token_size: int = 128
    preprocess_chunk_max_token_size: Optional[int] = None

    ie_model_name: str = "spacy"
    ie_batch_size: int = 256
    ieNer_device: int = 0
    ieNer_num_works: int = 4
    skip_graph: bool = False

    mllm_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    mllm_base_url: str = "http://localhost:8000/v1"

    embedding_model_name: str = "nvidia/NV-Embed-v2"
    embedding_batch_size: int = 16
    embedding_max_model_len: int = 45000
    embedding_return_as_normalized: bool = True
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = "auto"

    multimodal_embedding_model_name: Optional[str] = None
    multimodal_embedding_batch_size: int = 64
    multimodal_embedding_return_as_normalized: bool = True
    multimodal_embedding_max_model_len: int = 77
    multimodal_embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = "auto"

    clip_image_processor_name: Optional[str] = None
    clip_device: int = 0

    grounding_model_name: Optional[str] = None
    sam3_device: int = 0
    sam3_num_workers: int = 1
    grounding_model_work_dir: str = "samgrounding"
    grounding_model_prompt_batch_size: int = 64
    grounding_detection_threshold: float = 0.5

    synonymy_edge_topk: int = 2047
    synonymy_visual_edge_topk: int = 2047
    synonymy_edge_query_batch_size: int = 1000
    synonymy_edge_key_batch_size: int = 10000
    synonymy_edge_sim_threshold: float = 0.8
    synonymy_visual_edge_sim_threshold: float = 0.8
    is_directed_graph: bool = False


    text_sentence_top_k: int = 10
    text_entity_top_k: int = 10
    text_img_top_k: int = 10
    text_subimg_top_k: int = 10
    text_chunk_top_k: int = 4

    visual_sentence_top_k: int = 10
    visual_entity_top_k: int = 10
    visual_img_top_k: int = 20
    visual_subimg_top_k: int = 10
    visual_chunk_top_k: int = 10

    retrieval_top_k: int = 50
    ppr_batch_size: int = 128
    ppr_damping: float = 0.85
    ppr_max_iter: int = 100
    ppr_tolerance: float = 1e-6

    chunk_node_weight: float = 0.05
    img_node_weight: float = 1.0

    text_branch_weight: float = 1.0
    visual_branch_weight: float = 1.0
    
    ppr_device: int = 0
    retrieve_device: int = 0
    save_ieNer: bool = True
    max_qa_steps: int = 1

    qa_top_k: int = 1
    qa_context_max_len: int = 24000

    save_dir: Optional[str] = None

    dataset: Optional[Literal["hotpotqa", "hotpotqa_train", "musique", "2wikimultihopqa"]] = None

    graph_type: Literal[
        "dpr_only",
        "entity",
        "passage_entity",
        "relation_aware_passage_entity",
        "passage_entity_relation",
        "facts_and_sim_passage_node_unidirectional",
    ] = "facts_and_sim_passage_node_unidirectional"
    
    corpus_len: Optional[int] = None

    @classmethod
    def from_env(cls, prefix: str = "MG2RAG_", **overrides):
        """Create a config with any dataclass field overridable by environment.

        For example, ``ppr_damping`` maps to ``MG2RAG_PPR_DAMPING``. Explicit
        keyword overrides take precedence over environment values.
        """
        defaults = cls()
        values = {}
        for field in fields(cls):
            env_name = f"{prefix}{field.name.upper()}"
            if env_name not in os.environ:
                continue
            raw_value = os.environ[env_name].strip()
            current_value = getattr(defaults, field.name)
            values[field.name] = _parse_env_value(raw_value, field.type, current_value)
        values.update(overrides)
        return cls(**values)

    def __post_init__(self):
        if self.response_format is None:
            self.response_format = {"type": "json_object"}

        if self.save_dir is None:
            if self.dataset is None:
                logger.debug("No save_dir provided and no specific dataset running, defaulting save_dir to 'outputs'")
                self.save_dir = "outputs"
            else:
                self.save_dir = os.path.join("outputs", self.dataset)
        
        if self.sam3_device != self.ieNer_device:
            logger.info(f"Warning: grounding_model_device {self.sam3_device} is different from ieNer_device {self.ieNer_device}. Forcing them to be the same.")
            self.ieNer_device = self.sam3_device 
            
        logger.info(f"Initializing the highest level of save_dir to be {self.save_dir}")
    
    def __str__(self):
        config_items = vars(self)
        config_str = "\n".join(f"{key}: {value}" for key, value in config_items.items())
        return config_str


def _parse_env_value(raw_value, annotation, current_value):
    if raw_value.lower() in {"none", "null"}:
        return None
    target_type = type(current_value) if current_value is not None else None
    origin = get_origin(annotation)
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if target_type is None and origin is Union and args:
        target_type = args[0]
    elif target_type is None and origin is Literal and args:
        target_type = type(args[0])

    if target_type is bool:
        normalized = raw_value.lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean environment value: {raw_value}")
    if target_type is int:
        return int(raw_value)
    if target_type is float:
        return float(raw_value)
    if target_type in {dict, list} or origin in {dict, list, Dict, List}:
        return json.loads(raw_value)
    return raw_value
