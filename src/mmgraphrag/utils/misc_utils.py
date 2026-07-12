from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re

import os
from contextlib import contextmanager
import torch
from .logging_utils import get_logger
logger = get_logger(__name__)

from .typing import Triple
from .llm_utils import filter_invalid_triples

from PIL import Image

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]


@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal["node", "dpr"]


@dataclass
class QuerySolution:
    question: str
    image: Union[str, Image.Image, None]
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None

    def to_dict(self):
        image_val = self.image
        if isinstance(image_val, Image.Image):
            image_val = f"<PIL.Image mode={image_val.mode} size={image_val.size} id={id(image_val)}>"
        return {
            "question": self.question,
            "image": image_val,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": (
                [round(v, 4) for v in (self.doc_scores.tolist() if hasattr(self.doc_scores, 'tolist') else self.doc_scores)[:5]]
                if self.doc_scores is not None
                else None
            ),
            "gold_docs": self.gold_docs,
        }


def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()


def reformat_openie_results(
    corpus_openie_results,
) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):

    ner_output_dict = {
        chunk_item["idx"]: NerRawOutput(
            chunk_id=chunk_item["idx"],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item["extracted_entities"])),
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item["idx"]: TripleRawOutput(
            chunk_id=chunk_item["idx"],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item["extracted_triples"]),
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict


def extract_entity_nodes(
    chunk_triples: List[List[Triple]],
) -> (List[str], List[List[str]]):
    chunk_triple_entities = (
        []
    )
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(
                    f"During graph construction, invalid triple is found: {t}"
                )
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(
        np.unique([ent for ents in chunk_triple_entities for ent in ents])
    )
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[List[List[str]]]) -> List[Triple]:
    graph_triples = []
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples


def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val

    if range_val == 0:
        return np.ones_like(x)

    return (x - min_val) / range_val


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def compute_image_hash(image_input: Union[str, Image.Image], prefix: str = "") -> str:
    hash_md5 = md5()

    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"文件未找到: {image_input}")
            with open(image_input, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
                    
        elif isinstance(image_input, Image.Image):
            header = f"{image_input.mode}_{image_input.size[0]}x{image_input.size[1]}_"
            hash_md5.update(header.encode("utf-8"))
            
            try:
                img_array = np.asarray(image_input)
                
                if not img_array.flags['C_CONTIGUOUS']:
                    img_array = np.ascontiguousarray(img_array)
                    
                hash_md5.update(img_array)
                
            except Exception:
                hash_md5.update(image_input.tobytes())
            
        else:
            raise TypeError(f"不支持的类型: {type(image_input)}")

    except Exception as e:
        raise RuntimeError(f"Hash计算失败: {str(e)}")

    return prefix + hash_md5.hexdigest()

def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    value_iter = iter(data.values())

    try:
        first_length = len(next(value_iter))
    except StopIteration:
        return True

    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )

@contextmanager
def monitor_vram(device: str, task_name: str = "Model Loading"):
    """
    一个用于监控代码块显存变化的上下文管理器
    args:
        device: "cuda" or "cpu"
        task_name: 任务名称，用于日志输出
    usage:
        with monitor_vram("cuda", "Loading EVA-CLIP Model"):
            model = ...
    """
    mem_before = 0
    if device == "cuda":
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
        logger.debug(f" Start {task_name}... (Current VRAM: {mem_before / (1024**3):.2f} GB)")
    else:
        logger.debug(f" Start {task_name} on CPU...")

    try:
        yield
    finally:
        if device == "cuda":
            mem_after = torch.cuda.memory_allocated()
            model_mem_usage = mem_after - mem_before
            
            model_mem_gb = model_mem_usage / (1024**3)
            total_mem_gb = mem_after / (1024**3)

            logger.debug(f"  {task_name} Finished.")
            logger.debug(f"   - Delta VRAM Usage: {model_mem_gb:.2f} GB")
            logger.debug(f"   - Total VRAM Allocated: {total_mem_gb:.2f} GB")
        else:
            logger.debug(f"  {task_name} Finished on CPU.")

def get_mmlm_caption(text: str) -> str:
    """
    终极提取函数。
    可以处理：
    1. 标准带冒号: "**[Search Context]:** Content"
    2. 无冒号: "**[Search Context]** Content"
    3. 纯文本换行: "Search Context\nContent"
    4. 奇怪符号: "### Search Context - Content"
    """
    if not isinstance(text, str):
        logger.warning(f"[type]Expected string input for get_mmlm_caption, got {type(text)} : {text}")
        return None

    
    
    
    

    pattern = r"(?:^|\n)[\W_]*Search Context[\]\*\:\-\s]*(.*)"
    
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        logger.warning(f"[output no Search Context] No match found in text: {text}")
        return str(text).strip()


def clean_llm_text(text):
    if not text:
        return ""

    pattern = r"(?i)(?:\*\*|\[)?\s*(Visual Evidence|Search Context)\s*(?:\]|\*\*|)?\s*:?"
    
    cleaned_text = re.sub(pattern, "", text)
    
    return cleaned_text.strip()