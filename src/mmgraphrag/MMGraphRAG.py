import json
import os
import re
from transformers import AutoTokenizer
from dataclasses import asdict
from PIL import Image
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import cupy as cp
import cupyx.scipy.sparse as cps
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from tqdm import tqdm
import igraph as ig
import numpy as np
import time
from collections import ChainMap
import pickle
from .llm import _get_llm_class, BaseLLM, CacheMLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .grounding_model import SAM3Grounding, GroundingModel
from .embedding_store import MultimodalEmbeddingStore

from .information_extraction import SpacyNER
from .prompts.prompt_template_manager import PromptTemplateManager

from .utils.misc_utils import *
from .utils.config_utils import BaseConfig
from .utils.logging_utils import get_logger
import math
import gc

logger = get_logger(__name__)

import multiprocessing as mp
import traceback


def _isolated_grounding_worker(config, working_dir, tasks_data, queue, device_id):
    """
    完全隔离的 Grounding 子进程，由主进程动态分配显卡。
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            
        config.sam3_device = device_id
        config.grounding_device = device_id
            
        local_grounding_model = SAM3Grounding(
            global_config=config,
            global_work_dir=working_dir,
        )
        
        results_dict = {}
        from tqdm import tqdm
        for idx, img, doc_hashid, img_hashid, doc_entity_list in tqdm(tasks_data, desc="Isolated Grounding"):
            current_batch_size = config.grounding_model_prompt_batch_size
            results_grounding = None
            
            while results_grounding is None:
                try:
                    results_grounding = local_grounding_model.predict(
                        image=img, prompt=doc_entity_list, prompt_batch_size=current_batch_size
                    )
                except Exception as e:
                    is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (isinstance(e, RuntimeError) and "out of memory" in str(e).lower())
                    if is_oom:
                        torch.cuda.empty_cache()
                        old_bs = current_batch_size
                        current_batch_size = current_batch_size // 2
                        if current_batch_size < 1:
                            raise RuntimeError("Batch size reduced to 0. Cannot recover from OOM.")
                    else:
                        raise e 
                        
            subimg_data_list = local_grounding_model.get_save_crop(results=results_grounding, chunk_id=doc_hashid, img_id=img_hashid)
            sub_paths = [res.get("save_path", "") for res in subimg_data_list]
            labels = [res.get("label", "") for res in subimg_data_list]
            scores = [res.get("score", 1.0) for res in subimg_data_list]

            results_dict[idx] = (sub_paths, labels, scores)
            
        queue.put({"status": "success", "data": results_dict})
        
    except Exception as e:
        queue.put({"status": "error", "data": traceback.format_exc()})

def _isolated_ner_worker(config, input_data, queue):
    """在一个完全独立的进程中运行 NER，跑完即刻销毁，实现显存物理级释放"""
    try:
        local_ner = SpacyNER(global_config=config)
        result = local_ner.batch_ieNer(input_data)
        
        queue.put({"status": "success", "data": result})
    except Exception as e:
        import traceback
        queue.put({"status": "error", "data": traceback.format_exc()})

class MMGraphRAG:

    def __init__(self, global_config=None, only_qa=False):

        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.info(f"MMGraphRAG init with config:\n  {_print_config}\n")
        
        

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)
        self.mllm_model: BaseLLM = CacheMLLM(global_config=self.global_config)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.global_config.tokenizer_path, 
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Tokenizer init failed: {e}. QA truncation might fail if not fixed.")
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        if not only_qa:
            multimodal_embedding_label = self.global_config.multimodal_embedding_model_name.replace("/", "_")
            clean_label = re.sub(r"_data\d+_sijundai_models_", "", multimodal_embedding_label)
            self.working_dir = os.path.join(self.global_config.save_dir, clean_label)        
            if not os.path.exists(self.working_dir):
                logger.debug(f"Creating working directory: {self.working_dir}")
                os.makedirs(self.working_dir, exist_ok=True)

            self.ieNER = None
            self.graph = self.initialize_graph()

            tmp_multimodal_embedding_name = self.global_config.multimodal_embedding_model_name
            self.multimodal_embedding_model: BaseEmbeddingModel = _get_embedding_model_class(embedding_model_name=tmp_multimodal_embedding_name)(
                global_config=self.global_config,
                multimodal_embedding_model_name=self.global_config.multimodal_embedding_model_name,
            )
            self.grounding_model = None
            def _init_store(subdir, type_name):
                return MultimodalEmbeddingStore(
                    self.multimodal_embedding_model,
                    os.path.join(self.working_dir, subdir),
                    self.global_config.embedding_batch_size,
                    type_name,
                )

            with ThreadPoolExecutor() as executor:
                f_chunk = executor.submit(_init_store, "chunk_embeddings", "chunk")
                f_entity = executor.submit(_init_store, "entity_embeddings", "entity")
                f_sentence = executor.submit(_init_store, "sentence_embeddings", "sentence")
                f_image = executor.submit(_init_store, "image_embeddings", "image")
                f_subimg = executor.submit(_init_store, "subimg_embeddings", "subimg")

                self.chunk_embedding_store = f_chunk.result()
                self.entity_embedding_store = f_entity.result()
                self.sentence_embedding_store = f_sentence.result()
                self.image_embedding_store = f_image.result()
                self.subimg_embedding_store = f_subimg.result()

            self.ieNer_results_path = os.path.join(self.global_config.save_dir,f"ieNer_results.json")

            self.ready_to_retrieve = False
            self.ppr_time = 0
            self.rerank_time = 0
            self.all_retrieval_time = 0
            
            self.ent_node_to_chunk_ids = {}
            self.global_chunk_entities_map = {}
            self.global_chunk_sentences_map = {}
            self.global_chunk_triples_map = {}
            self.global_sentence_entities_map = {}
            self.subimg_to_entity_map = {}
            self.aux_data_path = os.path.join(self.global_config.save_dir, "aux_maps.pkl")
            self.load_aux_data()
            self.query_cache_path = os.path.join(self.working_dir, "query_embedding_cache.pkl")
        
        else:
            logger.info(">>> Initializing in QA-Only Mode (Skipping Retrieval Components) <<<")
            self.ieNER = None
            self.graph = None
            self.multimodal_embedding_model = None
            self.grounding_model = None
            self.chunk_embedding_store = None

    def load_aux_data(self):
        """ 加载辅助映射数据"""
        if os.path.exists(self.aux_data_path):
            try:
                with open(self.aux_data_path, "rb") as f:
                    data = pickle.load(f)
                self.ent_node_to_chunk_ids = data.get("ent_node_to_chunk_ids", {})
                self.global_chunk_entities_map = data.get("global_chunk_entities_map", {})
                self.global_chunk_sentences_map = data.get("global_chunk_sentences_map", {})
                self.global_chunk_triples_map = data.get("global_chunk_triples_map", {})
                self.global_sentence_entities_map = data.get("global_sentence_entities_map", {})
                logger.info("Loaded auxiliary data.")
            except Exception as e:
                logger.error(f"Failed to load aux data: {e}")

    def initialize_graph(self):
        self._graph_pickle_filename = os.path.join(self.global_config.save_dir, f"graph.pickle")
        preloaded_graph = None
        
        if not self.global_config.force_index_from_scratch:
            logger.info(f"Checking for existing graph pickle at {self._graph_pickle_filename}")
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)
        
        if preloaded_graph is None:
            self.graph = ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(f"Loaded graph with {preloaded_graph.vcount()} nodes.")
            self.graph = preloaded_graph

        self._update_node_indices()
        return self.graph

    def _update_node_indices(self):
        """
        更新图节点名称到索引的映射缓存。
        """
        if hasattr(self, "graph") and hasattr(self.graph, "vs"):
            try:
                self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs}
                logger.info(f"Updated node index map. Total nodes: {len(self.node_name_to_vertex_idx)}")
            except KeyError:
                logger.warning("Graph contains nodes without 'name' attribute. Index map might be incomplete.")
                self.node_name_to_vertex_idx = {}
        else:
            self.node_name_to_vertex_idx = {}

    def index(self, docs: List[str], imgs: List[Union[str, Image.Image]], match_arr: List[int]):
        logger.debug(f"Indexing Documents")
        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()
        existing_chunk_hash_id_to_ner_data, new_chunk_hash_ids = self.load_existing_ieNer(chunk_to_rows.keys())
        new_ieNer_rows = {k: chunk_to_rows[k]["content"] for k in new_chunk_hash_ids}
        
        if len(new_chunk_hash_ids) > 0:
            new_chunk_hash_id_to_ner_data = {}
            
            if self.global_config.ie_model_name is not None:
                num_workers = self.global_config.ieNer_num_works
                logger.info(f"Spawning {num_workers} isolated processes for NER ({len(new_ieNer_rows)} docs)...")
                
                items = list(new_ieNer_rows.items())
                chunk_size = math.ceil(len(items) / num_workers)
                sub_dicts = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
                
                ctx = mp.get_context('spawn')
                processes = []
                queues = []
                
                for i, sub_dict in enumerate(sub_dicts):
                    if not sub_dict: 
                        continue
                    q = ctx.Queue()
                    p = ctx.Process(target=_isolated_ner_worker, args=(self.global_config, sub_dict, q))
                    processes.append(p)
                    queues.append(q)
                    p.start()
                
                for i, q in enumerate(queues):
                    result_dict = q.get()
                    if result_dict["status"] == "error":
                        logger.error(f"NER Process {i} Failed:\n{result_dict['data']}")
                        for p in processes:
                            if p.is_alive(): p.terminate()
                        raise RuntimeError(f"NER execution failed in child process {i}.")
                    
                    new_chunk_hash_id_to_ner_data.update(result_dict["data"])
                
                for p in processes:
                    p.join()
                    
                logger.info("All NER Processes terminated successfully. VRAM is 100% physically reclaimed by the OS.")

            self.merge_ieNer_results(existing_chunk_hash_id_to_ner_data, new_chunk_hash_id_to_ner_data)
            if self.global_config.save_ieNer:
                self.save_ieNer_results(existing_chunk_hash_id_to_ner_data)

        self.node_to_node_stats = {}

        entity_nodes, sentence_nodes = self.update_graph_stats(existing_chunk_hash_id_to_ner_data)
        self.entity_embedding_store.insert_strings(entity_nodes)
        logger.info(f"entity nodes to insert: {len(entity_nodes)}, sentence nodes to insert: {len(sentence_nodes)}")
        self.sentence_embedding_store.insert_strings(sentence_nodes)

        has_image_updates = (imgs is not None) and (len(imgs) > 0)
        if has_image_updates:
            docs_ids, imgs_ids, subimgs_lists, ent_lists, scores_lists = self._process_multimodal_grounding(imgs, docs, match_arr)
            self.batch_add_multimodal_edges(docs_ids, imgs_ids, subimgs_lists, ent_lists, scores_lists)

        if len(new_chunk_hash_ids) > 0 or has_image_updates:
            logger.info(f"Graph updates detected. New Chunks: {len(new_chunk_hash_ids)}, New Images: {len(imgs) if imgs else 0}")
            self.augment_graph()
            self.save_igraph()
        else:
            logger.info("No new content (Text or Images) detected. Graph not updated.")
        
        self.ready_to_retrieve = False 
        logger.info("Indexing completed. Retrieval state reset.")

    def load_existing_ieNer(self, chunk_hash_ids: List[str]) -> Tuple[Dict[str, Any], Set[str]]:
        """
        Returns:
            "doc_id": {
                "entities": { "Apple": "ORG", ... },
                "sentences": { "Sentence...": ["Apple"], ... },
                "relations": [ {"head": "Steve", "relation": "founded", "tail": "Apple"}, ... ]
            },
            new_chunk_hash_ids: set(需要进行推理的 ID)
        """
        if os.path.exists(self.ieNer_results_path):
            try:
                with open(self.ieNer_results_path, 'r', encoding='utf-8') as f:
                    all_cached_data = json.load(f)
                cached_ids = set(all_cached_data.keys())
                target_ids = set(chunk_hash_ids)
                new_chunk_hash_ids = target_ids - cached_ids
                return all_cached_data, new_chunk_hash_ids
            except json.JSONDecodeError:
                logger.warning(f"现有文件 {self.ieNer_results_path} 格式错误，将重新处理所有数据。")
                return {}, set(chunk_hash_ids)
        else:
            return {}, set(chunk_hash_ids)

    def merge_ieNer_results(
        self,
        existing_chunk_hash_id_to_data: Dict[str, Dict[str, Any]],
        new_chunk_hash_id_to_data: Dict[str, Dict[str, Any]],
    )-> Dict[str, Dict[str, Any]]:
        """
        合并已有的 NER 结果和新推理得到的结果. input 与return有相同的结构
        :return: {
            "doc_id": {
                "entities": { "Apple": "ORG", ... },
                "sentences": { "Sentence...": ["Apple"], ... },
                "relations": [ {"head": "Steve", "relation": "founded", "tail": "Apple"}, ... ]
            },
            ...
        }
        """
        existing_chunk_hash_id_to_data.update(new_chunk_hash_id_to_data)
        return existing_chunk_hash_id_to_data

    def save_ieNer_results(self, existing_chunk_hash_id_to_data: Dict[str, Dict[str, Any]])-> None:
        with open(self.ieNer_results_path, "w", encoding="utf-8") as f:
            json.dump(existing_chunk_hash_id_to_data, f, ensure_ascii=False)
    
    def update_graph_stats(self, chunk_hash_id_to_data: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        根据统一的 NER 结果更新图统计信息 (共现权重 & 索引)。
        :param chunk_hash_id_to_data:
            {
            "doc_id": {
            "entities": { "Apple": "ORG", ... },
            "sentences": { "Sentence...": ["Apple"], ... },
            "relations": [ {"head": "Steve", "relation": "founded", "tail": "Apple"}, ... ]
            },
            ...
            }
        """
        collected_entities = set()
        collected_sentences = set()
        
        batch_chunk_entities = {}
        batch_chunk_sentences = {}
        batch_chunk_triples = {}
        batch_sentence_entities = {}

        current_graph_nodes = set()
        if hasattr(self, "graph") and hasattr(self.graph, "vs") and "name" in self.graph.vs.attributes():
            current_graph_nodes = set(self.graph.vs["name"])

        for cid, data in chunk_hash_id_to_data.items():
            raw_entities = data.get("entities", {})
            processed_entities = {text_processing(entity): label for entity, label in raw_entities.items()}
            raw_sentences = data.get("sentences", {})
            processed_sentences = {text_processing(sent): text_processing(ents) for sent, ents in raw_sentences.items()}
            raw_relations = data.get("relations", [])
            processed_relations = [
                {
                    "head": text_processing(r["head"]),
                    "relation": r["relation"], 
                    "tail": text_processing(r["tail"])
                }
                for r in raw_relations
            ]

            batch_chunk_entities[cid] = processed_entities
            batch_chunk_sentences[cid] = processed_sentences
            batch_chunk_triples[cid] = processed_relations
            batch_sentence_entities.update(processed_sentences)

            collected_entities.update(processed_entities.keys())
            collected_sentences.update(processed_sentences.keys())

            if cid in current_graph_nodes:
                continue

            for entity_name in processed_entities.keys():
                node_key = compute_mdhash_id(content=entity_name, prefix="entity-")
                if node_key not in self.ent_node_to_chunk_ids:
                    self.ent_node_to_chunk_ids[node_key] = set()
                self.ent_node_to_chunk_ids[node_key].add(cid)
                self.node_to_node_stats[(cid, node_key)] = (self.node_to_node_stats.get((cid, node_key), 0.0) + 1.0)

            for relation in processed_relations:
                head = relation["head"]
                tail = relation["tail"]
                if head and tail:
                    head_key = compute_mdhash_id(content=head, prefix="entity-")
                    tail_key = compute_mdhash_id(content=tail, prefix="entity-")
                    self.node_to_node_stats[(head_key, tail_key)] = self.node_to_node_stats.get((head_key, tail_key), 0.0) + 1.0
                    self.node_to_node_stats[(tail_key, head_key)] = self.node_to_node_stats.get((tail_key, head_key), 0.0) + 1.0

        self.global_chunk_entities_map.update(batch_chunk_entities)
        self.global_chunk_sentences_map.update(batch_chunk_sentences)
        self.global_chunk_triples_map.update(batch_chunk_triples)
        self.global_sentence_entities_map.update(batch_sentence_entities)

        return list(collected_entities), list(collected_sentences)
    
    def _process_multimodal_grounding(
        self, imgs: List[Union[str, Image.Image]], docs: List[str], match_arr: List[int]
    )-> Tuple[List[str], List[str], List[List[str]], List[List[str]], List[List[float]]]:
        """
        :return:
        docs_hashid_list, imgs_hashid_list, subimgs_hashid_lists, entity_hashid_lists, grounding_scores_lists
        """
        total_count = len(imgs)
        docs_hashid_list = [None] * total_count
        imgs_hashid_list = [None] * total_count
        processed_results = [None] * total_count
        all_subimg_paths_to_register = []

        for i, doc_idx in enumerate(match_arr):
            docs_hashid_list[i] = compute_mdhash_id(docs[doc_idx], prefix="chunk-")
        self.image_embedding_store.insert_inputs(inputs=imgs)
        for i, img in enumerate(imgs):
            imgs_hashid_list[i] = self.image_embedding_store.get_hash_id(img)


        target_grounding_work_dir = os.path.join(self.working_dir, self.global_config.grounding_model_work_dir) 

        inference_tasks = []

        def check_cache_single(idx):
            img_hashid = imgs_hashid_list[idx]
            doc_hashid = docs_hashid_list[idx]
            expected_cache_dir = os.path.join(target_grounding_work_dir, str(doc_hashid), str(img_hashid))
            metadata_path = os.path.join(expected_cache_dir, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    sub_paths = []
                    labels = []
                    scores = []
                    for obj in metadata.get("objects", []):
                        sub_paths.append(os.path.join(target_grounding_work_dir, obj["rel_path"]))
                        labels.append(obj["label"])
                        scores.append(obj["score"])
                    return idx, True, (sub_paths, labels, scores)
                except Exception:
                    pass
            return idx, False, None
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(check_cache_single, i) for i in range(total_count)]
            for future in futures:
                idx, is_hit, data = future.result()
                if is_hit:
                    processed_results[idx] = data
                    all_subimg_paths_to_register.extend(data[0])
                else:
                    inference_tasks.append((idx, imgs[idx], docs_hashid_list[idx], imgs_hashid_list[idx]))

        if inference_tasks:
            num_workers = self.global_config.sam3_num_workers
            base_device = getattr(self.global_config, 'sam3_device', 1)
            
            logger.info(f"Cache miss: {len(inference_tasks)}/{total_count}. Spawning {num_workers} isolated processes across GPUs...")

            tasks_data = []
            for idx, img, doc_hashid, img_hashid in inference_tasks:
                doc_entity_list = self.get_filtered_entities(doc_hashid=doc_hashid)
                tasks_data.append((idx, img, doc_hashid, img_hashid, doc_entity_list))
                
            chunk_size = math.ceil(len(tasks_data) / num_workers)
            tasks_chunks = [tasks_data[i:i + chunk_size] for i in range(0, len(tasks_data), chunk_size)]

            ctx = mp.get_context('spawn')
            processes = []
            queues = []

            for i, chunk in enumerate(tasks_chunks):
                if not chunk: continue
                
                current_device_id = base_device + i 
                
                q = ctx.Queue()
                p = ctx.Process(target=_isolated_grounding_worker, args=(self.global_config, self.working_dir, chunk, q, current_device_id))
                processes.append(p)
                queues.append(q)
                p.start()
                logger.info(f"Started Grounding Worker {i} on cuda:{current_device_id}")

            for i, q in enumerate(queues):
                result_dict = q.get()
                if result_dict["status"] == "error":
                    logger.error(f"Grounding Process {i} Failed:\n{result_dict['data']}")
                    for p in processes:
                        if p.is_alive(): p.terminate()
                    raise RuntimeError(f"Grounding execution failed in child process {i}.")

                worker_results = result_dict["data"]
                for idx, (sub_paths, labels, scores) in worker_results.items():
                    processed_results[idx] = (sub_paths, labels, scores)
                    all_subimg_paths_to_register.extend(sub_paths)

            for p in processes:
                p.join()
                
            logger.info("All Multi-GPU Grounding inference finished. VRAM is 100% physically clean.")
            
            if result_dict["status"] == "error":
                logger.error(f"Grounding Process Failed:\n{result_dict['data']}")
                raise RuntimeError("Grounding execution failed in child process. VRAM is released.")
                
            worker_results = result_dict["data"]
            for idx, (sub_paths, labels, scores) in worker_results.items():
                processed_results[idx] = (sub_paths, labels, scores)
                all_subimg_paths_to_register.extend(sub_paths)

            logger.info("Grounding inference finished via isolated process. VRAM is 100% physically clean.")

        if all_subimg_paths_to_register:
            logger.info(f"Batch inserting {len(all_subimg_paths_to_register)} sub-images...")
            self.subimg_embedding_store.insert_inputs(all_subimg_paths_to_register)

        subimgs_hashid_lists = []
        entity_hashid_lists = []
        grounding_scores_lists = []

        def get_ids_single_task(data_tuple):
            sub_paths, labels, scores = data_tuple
            current_subimg_ids = [self.subimg_embedding_store.get_hash_id(p) for p in sub_paths]
            current_entity_ids = [self.entity_embedding_store.get_hash_id(l) for l in labels]
            return current_subimg_ids, current_entity_ids, scores
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(get_ids_single_task, processed_results)
            for s_ids, e_ids, sc in results:
                subimgs_hashid_lists.append(s_ids)
                entity_hashid_lists.append(e_ids)
                grounding_scores_lists.append(sc)

        lengths = {
            "imgs": len(imgs_hashid_list),
            "docs": len(docs_hashid_list),
            "subimgs": len(subimgs_hashid_lists),
            "entities": len(entity_hashid_lists),
            "scores": len(grounding_scores_lists),
        }
        assert len(set(lengths.values())) == 1, f"Alignment Error: {lengths}"
        return docs_hashid_list, imgs_hashid_list, subimgs_hashid_lists, entity_hashid_lists, grounding_scores_lists

    def get_filtered_entities(self, doc_hashid: str) -> List[str]:
        """
        根据 doc_hashid 获取实体列表，过滤掉数字类型，并进行文本标准化清洗。
        """
        included_labels = {"FAC", "LOC", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"}
        entities_map = self.global_chunk_entities_map.get(doc_hashid, {})
        raw_entities = [text for text, label in entities_map.items() if label in included_labels]
        processed_entities = text_processing(raw_entities)
        return list(set(processed_entities))

    def batch_add_multimodal_edges(
        self, docs_ids: list[str], imgs_ids: list[str], 
        subimgs_lists: list[list[str]], entity_lists: list[list[str]], 
        scores_lists: list[list[float]]
    ):
        logger.info(f"Adding multimodal edges for {len(imgs_ids)} images.")
        is_directed = self.global_config.is_directed_graph
        def edge_exists(u_name, v_name):
            u_idx = self.node_name_to_vertex_idx.get(u_name)
            v_idx = self.node_name_to_vertex_idx.get(v_name)
            if u_idx is None or v_idx is None:
                return False
            eid = self.graph.get_eid(u_idx, v_idx, directed=is_directed, error=False)
            return eid != -1
        for doc_id, img_id, sub_ids, ent_ids, scores in zip(docs_ids, imgs_ids, subimgs_lists, entity_lists, scores_lists):
            if doc_id and img_id:
                if not edge_exists(doc_id, img_id):
                    self.node_to_node_stats[(doc_id, img_id)] = 1.0
            if not sub_ids: continue
            for sub_id, ent_id, score in zip(sub_ids, ent_ids, scores):
                
                if sub_id not in self.subimg_to_entity_map:
                    self.subimg_to_entity_map[sub_id] = set()
                self.subimg_to_entity_map[sub_id].add(ent_id)
                current_w = self.node_to_node_stats.get((img_id, ent_id), 0.0)
                self.node_to_node_stats[(img_id, ent_id)] = max(current_w, score)
                
                if is_directed:
                    if not edge_exists(ent_id, img_id):
                        self.node_to_node_stats[(ent_id, img_id)] = score

    def augment_graph(self):
        nodes_added = self.add_new_nodes()
        
        if nodes_added:
            self._update_node_indices()

        self.add_new_edges()
        logger.info(f"Graph construction completed!\n{self.get_graph_info()}")

    def add_new_nodes(self):

        existing_node_ids = {v["name"] for v in self.graph.vs if "name" in v.attributes()}
        node_to_rows = ChainMap(
            self.entity_embedding_store.get_all_id_to_rows(),
            self.chunk_embedding_store.get_all_id_to_rows(),
            self.image_embedding_store.get_all_id_to_rows(),
        )

        nodes_to_add = []
        all_keys = set(["name"])

        for node_id, node_data in node_to_rows.items():
            if node_id not in existing_node_ids:
                temp_node = node_data.copy()
                temp_node["name"] = node_id
                nodes_to_add.append(temp_node)
                all_keys.update(temp_node.keys())

        if nodes_to_add:
            count = len(nodes_to_add)
            attributes = {k: [] for k in all_keys}
            for node in nodes_to_add:
                for k in all_keys:
                    attributes[k].append(node.get(k, None))
            self.graph.add_vertices(n=count, attributes=attributes)
            logger.info(f"Added {count} new nodes.")
            return True
        return False

    def add_new_edges(self):
        logger.info(f"Processing {len(self.node_to_node_stats)} raw edges from buffer...")
        edges_to_add = []
        weights_to_add = []
        current_node_names = self.node_name_to_vertex_idx
        for (src, tgt), weight in self.node_to_node_stats.items():
            if src == tgt: continue
            
            if src in current_node_names and tgt in current_node_names:
                edges_to_add.append((src, tgt))
                weights_to_add.append(weight)
            else:
                logger.info(f"Skipping edge {src}->{tgt}, node missing.")
                pass
        if not edges_to_add:
            logger.info("No valid edges to add.")
            return

        self.graph.add_edges(edges_to_add, attributes={"weight": weights_to_add})
        logger.info(f"Added {len(edges_to_add)} raw edges. Now simplifying...")

        original_ecount = self.graph.ecount()
        self.graph.simplify(multiple=True, loops=True, combine_edges={"weight": "sum"})
        final_ecount = self.graph.ecount()
        logger.info(f"Graph simplified. Edges reduced from {original_ecount} (Raw) to {final_ecount} (Weighted Unique).")


    def get_graph_info(self) -> Dict:
        graph_info = {}

        ids_chunk = set(self.chunk_embedding_store.get_all_ids())
        ids_entity = set(self.entity_embedding_store.get_all_ids())
        ids_image = set(self.image_embedding_store.get_all_ids())
        ids_subimg = set(self.subimg_embedding_store.get_all_ids())
        
        graph_info["num_entity_nodes"] = len(ids_entity)
        graph_info["num_chunk_nodes"] = len(ids_chunk)
        graph_info["num_image_nodes"] = len(ids_image)
        graph_info["num_subimg"] = len(ids_subimg)
        graph_info["num_total_nodes"] = len(ids_entity) + len(ids_chunk) + len(ids_image)

        unique_text_pairs = set()
        for relations_list in self.global_chunk_triples_map.values():
            for relation in relations_list:
                h = relation.get("head")
                t = relation.get("tail")
                if h and t:
                    unique_text_pairs.add((h, t))

        graph_info["num_unique_fact_triples"] = len(unique_text_pairs)

        stats = {
            "entity_entity": 0,
            "chunk_entity": 0,
            "chunk_image": 0,
            "image_entity": 0,
            "others": 0
        }

        def get_node_type(node_id):
            if node_id.startswith("chunk-"): return "chunk"
            if node_id.startswith("entity-"): return "entity"
            if node_id in ids_chunk: return "chunk"
            if node_id in ids_entity: return "entity"
            if node_id in ids_image: return "image"
            if node_id in ids_subimg: return "image"
            return "unknown"

        for u, v in self.node_to_node_stats.keys():
            t_u = get_node_type(u)
            t_v = get_node_type(v)
            
            edge_type_pair = tuple(sorted([t_u, t_v]))
            
            if edge_type_pair == ("entity", "entity"):
                stats["entity_entity"] += 1
            elif edge_type_pair == ("chunk", "entity"):
                stats["chunk_entity"] += 1
            elif edge_type_pair == ("chunk", "image"):
                stats["chunk_image"] += 1
            elif edge_type_pair == ("entity", "image"):
                stats["image_entity"] += 1
            else:
                stats["others"] += 1

        logger.info(f"Edge Type Stats updated: {stats}")
        graph_info["num_entity_entity_edges"] = stats["entity_entity"]
        graph_info["num_chunk_entity_edges"] = stats["chunk_entity"]
        graph_info["num_chunk_image_edges"] = stats["chunk_image"]
        graph_info["num_image_entity_edges"] = stats["image_entity"]
        graph_info["num_raw_total_edges"] = len(self.node_to_node_stats)

        if hasattr(self, "graph") and self.graph is not None:
            graph_info["num_final_graph_edges"] = self.graph.ecount()
        else:
            graph_info["num_final_graph_edges"] = 0
            
        return graph_info
    
    def save_igraph(self):
        self.graph.write_pickle(self._graph_pickle_filename)
        self.save_aux_data()
        logger.info(f"Completed Graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges")

    def save_aux_data(self):
        """保存辅助映射数据到磁盘"""
        data = {
            "ent_node_to_chunk_ids": self.ent_node_to_chunk_ids,
            "global_chunk_entities_map": self.global_chunk_entities_map,
            "global_chunk_sentences_map": self.global_chunk_sentences_map,
            "global_chunk_triples_map": self.global_chunk_triples_map,
            "global_sentence_entities_map": self.global_sentence_entities_map,
            "subimg_to_entity_map": self.subimg_to_entity_map
        }
        with open(self.aux_data_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved auxiliary data to {self.aux_data_path}")

    def retrieve(self, question_queries: List[str], image_queries: List[Union[str, Image.Image]] = None) -> List[QuerySolution]:
        """
        [Optimized] 批量检索函数：支持单模态/多模态混合 Batch
        Batch Embedding -> Batch FAISS -> Batch Seed Matrix (Weighted) -> Batch PPR
        """
        retrieve_start_time = time.time()
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        
        self.get_question_query_embeddings(question_queries)
        self.get_image_query_embeddings(image_queries)
        
        num_queries = len(question_queries)
        
        q_text_embeds_list = []
        q_img_embeds_list = []
        
        for i in range(num_queries):
            q_text = question_queries[i]
            if q_text and isinstance(q_text, str) and q_text.strip():
                q_text_embeds_list.append(self.query_to_embedding["textual"].get(q_text))
            else:
                q_text_embeds_list.append(None)
            
            curr_img = None
            if image_queries and i < len(image_queries):
                curr_img = image_queries[i]
            
            if curr_img:
                key = compute_image_hash(image_input=curr_img, prefix="imgquery-")
                q_img_embeds_list.append(self.query_to_embedding["visual"].get(key))
            else:
                q_img_embeds_list.append(None)

        
        search_results = {i: {} for i in range(num_queries)}

        t_valid_idx = [i for i, e in enumerate(q_text_embeds_list) if e is not None]
        if t_valid_idx:
            batch_t_emb = np.vstack([q_text_embeds_list[i] for i in t_valid_idx])
            
            t2s_ids, t2s_sc = self._batch_torch_search(
                self.sentence_emb_T, batch_t_emb, self.global_config.text_sentence_top_k
            )
            t2c_ids, t2c_sc = self._batch_torch_search(
                self.chunk_emb_T, batch_t_emb, self.global_config.text_chunk_top_k
            )
            t2i_ids, t2i_sc = self._batch_torch_search(
                self.image_emb_T, batch_t_emb, self.global_config.text_img_top_k
            )
            t2sub_ids, t2sub_sc = self._batch_torch_search(
                self.subimg_emb_T, batch_t_emb, self.global_config.text_subimg_top_k
            )
            
            for ptr, q_idx in enumerate(t_valid_idx):
                search_results[q_idx]["t2s"] = (t2s_ids[ptr], t2s_sc[ptr])
                search_results[q_idx]["t2chunk"] = (t2c_ids[ptr], t2c_sc[ptr])
                search_results[q_idx]["t2img"] = (t2i_ids[ptr], t2i_sc[ptr])
                search_results[q_idx]["t2sub"] = (t2sub_ids[ptr], t2sub_sc[ptr])

        i_valid_idx = [i for i, e in enumerate(q_img_embeds_list) if e is not None]
        if i_valid_idx:
            batch_i_emb = np.vstack([q_img_embeds_list[i] for i in i_valid_idx])
            
            i2s_ids, i2s_sc = self._batch_torch_search(
                self.sentence_emb_T, batch_i_emb, self.global_config.visual_sentence_top_k
            )
            i2c_ids, i2c_sc = self._batch_torch_search(
                self.chunk_emb_T, batch_i_emb, self.global_config.visual_chunk_top_k
            )
            i2i_ids, i2i_sc = self._batch_torch_search(
                self.image_emb_T, batch_i_emb, self.global_config.visual_img_top_k
            )
            i2sub_ids, i2sub_sc = self._batch_torch_search(
                self.subimg_emb_T, batch_i_emb, self.global_config.visual_subimg_top_k
            )

            for ptr, q_idx in enumerate(i_valid_idx):
                search_results[q_idx]["i2s"] = (i2s_ids[ptr], i2s_sc[ptr])
                search_results[q_idx]["i2chunk"] = (i2c_ids[ptr], i2c_sc[ptr])
                search_results[q_idx]["i2img"] = (i2i_ids[ptr], i2i_sc[ptr])
                search_results[q_idx]["i2sub"] = (i2sub_ids[ptr], i2sub_sc[ptr])

        seed_rows = []
        seed_cols = []
        seed_data = []

        def apply_min_max_and_topk(node_score_map, top_k):
            if not node_score_map: return {}
            avg_scores = {idx: tot/cnt for idx, (tot, cnt) in node_score_map.items()}
            vals = list(avg_scores.values())
            if not vals: return {}
            v_min, v_max = min(vals), max(vals)
            denom = v_max - v_min + 1e-9
            
            normed_scores = []
            normed_indices = []
            for idx, val in avg_scores.items():
                normed_scores.append((val - v_min) / denom)
                normed_indices.append(idx)
            
            normed_scores = np.array(normed_scores)
            normed_indices = np.array(normed_indices)
            if len(normed_scores) > top_k:
                top_k_args = np.argsort(normed_scores)[-top_k:]
                return dict(zip(normed_indices[top_k_args], normed_scores[top_k_args]))
            else:
                return dict(zip(normed_indices, normed_scores))

        for q_idx in tqdm(range(num_queries), desc="Building Seeds"):
            curr_res = search_results[q_idx]
            
            text_branch_weights = {}
            image_branch_weights = {}
            
            def add_to_branch(branch_map, node_idx, w):
                if node_idx is not None:
                    branch_map[node_idx] = branch_map.get(node_idx, 0.0) + w

            if q_text_embeds_list[q_idx] is not None:
                if "t2s" in curr_res:
                    s_indices, s_scores = curr_res["t2s"]
                    t_ent_stats = {} 
                    if len(s_indices) > 0:
                        for s_faiss_idx, s_score in zip(s_indices, s_scores):
                            s_hash_id = self.sentence_keys[s_faiss_idx]
                            row = self.sentence_embedding_store.get_row(s_hash_id)
                            if row:
                                ents = self.global_sentence_entities_map.get(row.get("content", ""), [])
                                for ent in ents:
                                    ent_key = compute_mdhash_id(content=ent, prefix="entity-")
                                    ent_idx = self.node_name_to_vertex_idx.get(ent_key)
                                    if ent_idx is not None:
                                        freq = len(self.ent_node_to_chunk_ids.get(ent_key, set()))
                                        weighted_s = s_score / max(1, freq)
                                        if ent_idx not in t_ent_stats: t_ent_stats[ent_idx] = [0.0, 0]
                                        t_ent_stats[ent_idx][0] += weighted_s
                                        t_ent_stats[ent_idx][1] += 1
                    
                    for idx, val in apply_min_max_and_topk(t_ent_stats, self.global_config.text_entity_top_k).items():
                        add_to_branch(text_branch_weights, idx, val)

                if "t2chunk" in curr_res:
                    indices, scores = curr_res["t2chunk"]
                    if len(indices) > 0:
                        for i, score in zip(indices, scores):
                            k = self.chunk_node_keys[i]
                            add_to_branch(text_branch_weights, self.node_name_to_vertex_idx.get(k), score * self.global_config.chunk_node_weight)
                
                if "t2img" in curr_res:
                    indices, scores = curr_res["t2img"]
                    if len(indices) > 0:
                        for i, score in zip(indices, scores):
                            k = self.image_node_keys[i]
                            add_to_branch(text_branch_weights, self.node_name_to_vertex_idx.get(k), score * self.global_config.img_node_weight)
                
                if "t2sub" in curr_res:
                    indices, scores = curr_res["t2sub"]
                    if len(indices) > 0:
                        for i, score in zip(indices, scores):
                            sub_k = self.subimg_node_keys[i]
                            
                            ent_k_set = self.subimg_to_entity_map.get(sub_k)
                            if ent_k_set:
                                for ent_k in ent_k_set:
                                    ent_idx = self.node_name_to_vertex_idx.get(ent_k)
                                    if ent_idx is not None:
                                        add_to_branch(text_branch_weights, ent_idx, score)

            if q_img_embeds_list[q_idx] is not None:
                if "i2s" in curr_res:
                    s_indices, s_scores = curr_res["i2s"]
                    i_ent_stats = {} 
                    if len(s_indices) > 0:
                        for s_faiss_idx, s_score in zip(s_indices, s_scores):
                            s_hash_id = self.sentence_keys[s_faiss_idx]
                            row = self.sentence_embedding_store.get_row(s_hash_id)
                            if row:
                                ents = self.global_sentence_entities_map.get(row.get("content", ""), [])
                                for ent in ents:
                                    ent_key = compute_mdhash_id(content=ent, prefix="entity-")
                                    ent_idx = self.node_name_to_vertex_idx.get(ent_key)
                                    if ent_idx is not None:
                                        freq = len(self.ent_node_to_chunk_ids.get(ent_key, set()))
                                        weighted_s = s_score / max(1, freq)
                                        if ent_idx not in i_ent_stats: i_ent_stats[ent_idx] = [0.0, 0]
                                        i_ent_stats[ent_idx][0] += weighted_s
                                        i_ent_stats[ent_idx][1] += 1

                    for idx, val in apply_min_max_and_topk(i_ent_stats, self.global_config.visual_entity_top_k).items():
                        add_to_branch(image_branch_weights, idx, val)

                if "i2chunk" in curr_res:
                    indices, scores = curr_res["i2chunk"]
                    if len(indices) > 0:
                        for i, score in zip(indices, scores):
                            k = self.chunk_node_keys[i]
                            add_to_branch(image_branch_weights, self.node_name_to_vertex_idx.get(k), score * self.global_config.chunk_node_weight)
                
                if "i2img" in curr_res:
                    indices, scores = curr_res["i2img"]
                    if len(indices) > 0:
                        for i, score in zip(indices, scores):
                            k = self.image_node_keys[i]
                            add_to_branch(image_branch_weights, self.node_name_to_vertex_idx.get(k), score * self.global_config.img_node_weight)
                if "i2sub" in curr_res:
                    indices, scores = curr_res["i2sub"]
                    if len(indices) > 0:
                        for i, score in zip(indices, scores):
                            sub_k = self.subimg_node_keys[i]
                            ent_k_set = self.subimg_to_entity_map.get(sub_k)
                            if ent_k_set:
                                for ent_k in ent_k_set:
                                    ent_idx = self.node_name_to_vertex_idx.get(ent_k)
                                    if ent_idx is not None:
                                        add_to_branch(image_branch_weights, ent_idx, score)

            final_query_weights = {}
            
            if text_branch_weights:
                for node_idx, weight in text_branch_weights.items():
                    final_query_weights[node_idx] = final_query_weights.get(node_idx, 0.0) + (weight * self.global_config.text_branch_weight)
                
            if image_branch_weights:
                for node_idx, weight in image_branch_weights.items():
                    final_query_weights[node_idx] = final_query_weights.get(node_idx, 0.0) + (weight * self.global_config.visual_branch_weight)

            total_w = sum(final_query_weights.values())
            if total_w > 0:
                for n_idx, w in final_query_weights.items():
                    seed_rows.append(q_idx)
                    seed_cols.append(n_idx)
                    seed_data.append(w / total_w)
        
        if not seed_data:
            logger.warning("No seed nodes found for the batch. Returning empty results.")
            return [QuerySolution(question=q, image=(image_queries[i] if image_queries and i<len(image_queries) else None), docs=[], doc_scores=[]) for i, q in enumerate(question_queries)]

        num_nodes = self.graph.vcount()
        batch_seeds_csr = sp.coo_matrix(
            (seed_data, (seed_rows, seed_cols)), 
            shape=(num_queries, num_nodes)
        ).tocsr()

        ppr_start = time.time()
        logger.info(f"Running Batch PPR for {num_queries} queries...")
        
        chunk_scores_dense = self.run_batch_ppr(
            batch_seeds_csr, 
            target_indices=self.chunk_node_idxs
        )
        self.ppr_time += (time.time() - ppr_start)

        retrieval_results = []
        for q_idx in range(num_queries):
            scores = chunk_scores_dense[q_idx]
            
            sorted_indices_local = np.argsort(scores)[::-1]
            top_k_local = sorted_indices_local[:self.global_config.retrieval_top_k]
            
            top_docs = []
            top_scores = []
            
            for local_idx in top_k_local:
                doc_hash_id = self.chunk_node_keys[local_idx] 
                
                row = self.chunk_embedding_store.get_row(doc_hash_id)
                if row:
                    top_docs.append(row["content"])
                    top_scores.append(float(scores[local_idx]))
            
            curr_img_res = None
            if image_queries and q_idx < len(image_queries):
                curr_img_res = image_queries[q_idx]

            retrieval_results.append(QuerySolution(
                question=question_queries[q_idx],
                image=curr_img_res,
                docs=top_docs,
                doc_scores=top_scores
            ))

        self.all_retrieval_time += time.time() - retrieve_start_time
        logger.info(f"Retrieval done. Total: {self.all_retrieval_time:.2f}s (PPR: {self.ppr_time:.2f}s)")
        return retrieval_results

    def only_text_query_retrieve(self, question_queries: List[str]) -> List[QuerySolution]:
        """
        [Baseline] 仅使用文本 Query 进行 Dense Passage Retrieval (DPR)。
        不使用图结构 (PPR)，直接计算 Text Query Embedding 与 Chunk Embedding 的余弦相似度。
        """
        retrieve_start_time = time.time()
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_question_query_embeddings(question_queries)
        
        q_text_embeds_list = []
        valid_indices = []
        
        for i, q_text in enumerate(question_queries):
            if q_text and isinstance(q_text, str) and q_text.strip():
                emb = self.query_to_embedding["textual"].get(q_text)
                if emb is not None:
                    q_text_embeds_list.append(emb)
                    valid_indices.append(i)
        
        retrieval_results = [None] * len(question_queries)

        if q_text_embeds_list:
            batch_t_emb = np.vstack(q_text_embeds_list)
            indices_list, scores_list = self._batch_torch_search(
                self.chunk_emb_T, batch_t_emb, self.global_config.retrieval_top_k
            )

            for ptr, q_idx in enumerate(valid_indices):
                top_indices = indices_list[ptr]
                top_scores = scores_list[ptr]
                
                top_docs = []
                final_scores = []
                
                for local_idx, score in zip(top_indices, top_scores):
                    doc_hash_id = self.chunk_node_keys[local_idx]
                    row = self.chunk_embedding_store.get_row(doc_hash_id)
                    if row:
                        top_docs.append(row["content"])
                        final_scores.append(float(score))
                
                retrieval_results[q_idx] = QuerySolution(
                    question=question_queries[q_idx],
                    image=None,
                    docs=top_docs,
                    doc_scores=final_scores
                )

        for i in range(len(question_queries)):
            if retrieval_results[i] is None:
                retrieval_results[i] = QuerySolution(
                    question=question_queries[i], image=None, docs=[], doc_scores=[]
                )

        logger.info(f"Text-Only Baseline Retrieval done. Total: {time.time() - retrieve_start_time:.2f}s")
        return retrieval_results
    def only_img_retrievel(self, image_queries: List[Union[str, Image.Image]]) -> List[QuerySolution]:
        """
        [Baseline] 仅使用图像 Query 进行 Dense Passage Retrieval (DPR)。
        不使用图结构 (PPR)，直接计算 Image Query Embedding 与 Chunk Embedding 的余弦相似度。
        注意：这里返回的 QuerySolution 中 question 字段仅为占位符。
        """
        retrieve_start_time = time.time()
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_image_query_embeddings(image_inputs=image_queries)

        q_img_embeds_list = []
        valid_indices = []

        for i, img_input in enumerate(image_queries):
            if img_input:
                key = compute_image_hash(image_input=img_input, prefix="imgquery-")
                emb = self.query_to_embedding["visual"].get(key)
                if emb is not None:
                    q_img_embeds_list.append(emb)
                    valid_indices.append(i)

        retrieval_results = [None] * len(image_queries)

        if q_img_embeds_list:
            batch_i_emb = np.vstack(q_img_embeds_list)
            indices_list, scores_list = self._batch_torch_search(
                self.chunk_emb_T, batch_i_emb, self.global_config.retrieval_top_k
            )

            for ptr, q_idx in enumerate(valid_indices):
                top_indices = indices_list[ptr]
                top_scores = scores_list[ptr]
                
                top_docs = []
                final_scores = []
                
                for local_idx, score in zip(top_indices, top_scores):
                    doc_hash_id = self.chunk_node_keys[local_idx]
                    row = self.chunk_embedding_store.get_row(doc_hash_id)
                    if row:
                        top_docs.append(row["content"])
                        final_scores.append(float(score))
                
                retrieval_results[q_idx] = QuerySolution(
                    question="[Image Query]",
                    image=image_queries[q_idx],
                    docs=top_docs,
                    doc_scores=final_scores
                )

        for i in range(len(image_queries)):
            if retrieval_results[i] is None:
                retrieval_results[i] = QuerySolution(
                    question="[Image Query]", image=image_queries[i], docs=[], doc_scores=[]
                )

        logger.info(f"Image-Only Baseline Retrieval done. Total: {time.time() - retrieve_start_time:.2f}s")
        return retrieval_results

    def retrieve_via_visual_pivot(
        self, 
        queries: List[Union[str, Image.Image]], 
        query_type: Literal["text", "image"] = "text"
    ) -> List[QuerySolution]:
        """
        [New Strategy] 视觉透视检索：
        1. Query -> 检索库中的相似图片 (Top-K Images)
        2. Graph -> 查找这些图片连接的文本块 (Image -> Chunk)
        3. 返回对应的文本
        """
        retrieve_start_time = time.time()
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        chunk_node_idxs_set = set(self.chunk_node_idxs)

        if query_type == "text":
            self.get_question_query_embeddings(queries)
            valid_embeds = []
            valid_indices = []
            for i, q in enumerate(queries):
                if q and isinstance(q, str):
                    emb = self.query_to_embedding["textual"].get(q)
                    if emb is not None:
                        valid_embeds.append(emb)
                        valid_indices.append(i)
        else:
            self.get_image_query_embeddings(queries)
            valid_embeds = []
            valid_indices = []
            for i, q in enumerate(queries):
                if q:
                    key = compute_image_hash(image_input=q, prefix="imgquery-")
                    emb = self.query_to_embedding["visual"].get(key)
                    if emb is not None:
                        valid_embeds.append(emb)
                        valid_indices.append(i)

        retrieval_results = [None] * len(queries)

        if valid_embeds and self.image_emb_T is not None:
            batch_emb = np.vstack(valid_embeds)
            indices_list, scores_list = self._batch_torch_search(
                self.image_emb_T, batch_emb, self.global_config.retrieval_top_k
            )

            for ptr, q_idx in enumerate(valid_indices):
                top_img_indices = indices_list[ptr]
                top_img_scores = scores_list[ptr]
                
                found_docs_map = {} 

                for local_idx, score in zip(top_img_indices, top_img_scores):
                    img_hash_id = self.image_node_keys[local_idx]
                    
                    img_vertex_idx = self.node_name_to_vertex_idx.get(img_hash_id)
                    
                    if img_vertex_idx is not None:
                        neighbors = self.graph.neighbors(img_vertex_idx)
                        
                        for neighbor_idx in neighbors:
                            if neighbor_idx in chunk_node_idxs_set:
                                chunk_hash_id = self.graph.vs[neighbor_idx]["name"]
                                
                                if chunk_hash_id not in found_docs_map or score > found_docs_map[chunk_hash_id]["score"]:
                                    row = self.chunk_embedding_store.get_row(chunk_hash_id)
                                    if row:
                                        found_docs_map[chunk_hash_id] = {
                                            "content": row["content"],
                                            "score": float(score),
                                            "source_img": img_hash_id
                                        }

                sorted_docs = sorted(found_docs_map.values(), key=lambda x: x["score"], reverse=True)
                top_docs = [item["content"] for item in sorted_docs]
                final_scores = [item["score"] for item in sorted_docs]

                retrieval_results[q_idx] = QuerySolution(
                    question=queries[q_idx] if query_type == "text" else "[Image Query]",
                    image=queries[q_idx] if query_type == "image" else None,
                    docs=top_docs,
                    doc_scores=final_scores
                )

        for i in range(len(queries)):
            if retrieval_results[i] is None:
                retrieval_results[i] = QuerySolution(
                    question=queries[i] if query_type == "text" else "[Image Query]",
                    image=queries[i] if query_type == "image" else None,
                    docs=[], 
                    doc_scores=[]
                )

        logger.info(f"Visual Pivot Retrieval ({query_type} -> Image -> Text) done. Total: {time.time() - retrieve_start_time:.2f}s")
        return retrieval_results
    
    def prepare_retrieval_objects(self):

        logger.info("Preparing for retrieval.")
        if os.path.exists(self.query_cache_path):
            try:
                with open(self.query_cache_path, "rb") as f:
                    self.query_to_embedding = pickle.load(f)
                logger.info(f"Loaded query embedding cache from {self.query_cache_path}")
            except Exception as e:
                logger.error(f"Failed to load query cache: {e}. Initializing empty cache.")
                self.query_to_embedding = {"textual": {}, "visual": {}}
        else:
            self.query_to_embedding = {"textual": {}, "visual": {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids())
        self.chunk_node_keys: List = list(self.chunk_embedding_store.get_all_ids())
        self.sentence_keys: List = list(self.sentence_embedding_store.get_all_ids())
        self.image_node_keys = list(self.image_embedding_store.get_all_ids())
        self.subimg_node_keys = list(self.subimg_embedding_store.get_all_ids())

        expected_node_count = len(self.entity_node_keys) + len(self.chunk_node_keys) + len(self.image_node_keys)
        actual_node_count = self.graph.vcount()
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            if actual_node_count < expected_node_count:
                logger.info(f"Initializing/Updating graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
            self.node_name_to_vertex_idx = igraph_name_to_idx
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_chunk_nodes = [node_key for node_key in self.chunk_node_keys if node_key not in igraph_name_to_idx]
            missing_image_nodes = [node_key for node_key in self.image_node_keys if node_key not in igraph_name_to_idx]

            if any([missing_entity_nodes, missing_chunk_nodes, missing_image_nodes]):
                logger.warning(
                    f"Missing nodes in graph: "
                    f"Entity={len(missing_entity_nodes)}, chunk={len(missing_chunk_nodes)}, "
                    f"Image={len(missing_image_nodes)}"
                )
                self.add_new_nodes()
                self.save_igraph()
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx

            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys]
            self.chunk_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.chunk_node_keys]
            self.image_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.image_node_keys]
            self.subimg_node_idxs = []

        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.chunk_node_idxs = []
            self.image_node_idxs = []
            self.subimg_node_idxs = []

        logger.info("Loading embeddings and moving to GPU...")
        
        with ThreadPoolExecutor() as executor:
            f_entity = executor.submit(self.entity_embedding_store.get_embeddings, self.entity_node_keys)
            f_chunk = executor.submit(self.chunk_embedding_store.get_embeddings, self.chunk_node_keys)
            f_sentence = executor.submit(self.sentence_embedding_store.get_embeddings, self.sentence_keys)
            f_image = executor.submit(self.image_embedding_store.get_embeddings, self.image_node_keys)
            f_subimg = executor.submit(self.subimg_embedding_store.get_embeddings, self.subimg_node_keys)

            entity_embeddings_np = f_entity.result()
            chunk_embeddings_np = f_chunk.result()
            sentence_embeddings_np = f_sentence.result()
            image_embeddings_np = f_image.result()
            subimg_embeddings_np = f_subimg.result()


        
        self.chunk_emb_T = self._prepare_search_tensor(chunk_embeddings_np, self.global_config.retrieve_device)
        self.sentence_emb_T = self._prepare_search_tensor(sentence_embeddings_np, self.global_config.retrieve_device)
        self.image_emb_T = self._prepare_search_tensor(image_embeddings_np,  self.global_config.retrieve_device)
        self.subimg_emb_T = self._prepare_search_tensor(subimg_embeddings_np, self.global_config.retrieve_device)
        
        del entity_embeddings_np, chunk_embeddings_np, sentence_embeddings_np, image_embeddings_np, subimg_embeddings_np
        
        self.ready_to_retrieve = True



    def _prepare_search_tensor(self, embeddings: np.ndarray, device: str) -> torch.Tensor | None:
        """
        [New Helper] 将 numpy 向量转为 GPU 上的归一化转置 Tensor，用于矩阵乘法。
        Shape: [Dim, N]
        """
        if embeddings is None or embeddings.size == 0:
            return None
        
        t = torch.from_numpy(embeddings).to(dtype=torch.float32, device=device)
        
        t = torch.nn.functional.normalize(t, p=2, dim=1)
        
        return t.t().contiguous()

    def _batch_torch_search(self, target_tensor_T: torch.Tensor | None, query_embeddings: np.ndarray, top_k: int) -> Tuple[List[List[int]], List[List[float]]]:
        """
        [New Helper] 使用 PyTorch 矩阵乘法进行暴力检索 (Brute-Force Search)。
        :param target_tensor_T: 底库向量转置 Tensor, Shape [Dim, N], 已经在 GPU 上
        :param query_embeddings: 查询向量 Numpy, Shape [B, Dim]
        """
        if query_embeddings is None or len(query_embeddings) == 0:
            return [], []

        if target_tensor_T is None or target_tensor_T.shape[1] == 0:
            empty_rows = [[] for _ in range(len(query_embeddings))]
            return empty_rows, [row.copy() for row in empty_rows]

        device = target_tensor_T.device
        num_targets = target_tensor_T.shape[1]
        real_k = min(top_k, num_targets)

        with torch.no_grad():
            q_tensor = torch.from_numpy(query_embeddings).to(dtype=torch.float32, device=device)
            q_tensor = torch.nn.functional.normalize(q_tensor, p=2, dim=1)

            scores = torch.matmul(q_tensor, target_tensor_T)

            batch_scores_tensor, batch_indices_tensor = torch.topk(scores, k=real_k, dim=1, largest=True, sorted=True)

            indices_list = batch_indices_tensor.cpu().tolist()
            scores_list = batch_scores_tensor.cpu().tolist()

        normalized_scores = []
        for s_list in scores_list:
            if s_list:
                normalized_scores.append(min_max_normalize(s_list))
            else:
                normalized_scores.append([])

        return indices_list, normalized_scores
    
    def _save_query_cache(self):
        try:
            with open(self.query_cache_path, "wb") as f:
                pickle.dump(self.query_to_embedding, f)
            logger.info(f"Saved query embedding cache to {self.query_cache_path}")
        except Exception as e:
            logger.error(f"Failed to save query cache: {e}")

    def get_question_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and query.question not in self.query_to_embedding["textual"]:
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding["textual"]:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_sentence.")
            query_embeddings_for_sentence = self.multimodal_embedding_model.batch_encode(all_query_strings)
            for query, embedding in zip(all_query_strings, query_embeddings_for_sentence):
                self.query_to_embedding["textual"][query] = embedding
            self._save_query_cache()

    def get_image_query_embeddings(self, image_inputs: List[Union[str, Image.Image]]):
        batch_keys = []
        batch_pil_images = []
        for inp in image_inputs:
            if not inp:
                continue

            key = compute_image_hash(image_input=inp, prefix="imgquery-")
            if key in self.query_to_embedding["visual"]:
                continue

            img_obj = None
            try:
                if isinstance(inp, str):
                    if os.path.exists(inp):
                        img_obj = Image.open(inp).convert("RGB")
                    else:
                        logger.warning(f"Image path not found: {inp}")
                        continue
                elif isinstance(inp, Image.Image):
                    img_obj = inp.convert("RGB")
                else:
                    logger.warning(f"Unsupported input type: {type(inp)}")
                    continue
            except Exception as e:
                logger.error(f"Failed to load image input: {e}")
                continue
            if img_obj is not None:
                batch_keys.append(key)
                batch_pil_images.append(img_obj)

        if len(batch_pil_images) > 0:
            logger.info(f"Encoding {len(batch_pil_images)} new image queries...")
            try:
                image_embeddings_np = self.multimodal_embedding_model.batch_encode(images=batch_pil_images)
                count = 0
                for key, embedding_row in zip(batch_keys, image_embeddings_np):
                    self.query_to_embedding["visual"][key] = embedding_row
                    count += 1
                logger.info(f"Cached {count} visual embeddings.")
            except Exception as e:
                logger.error(f"Error executing batch encoding: {e}")
            self._save_query_cache()
        else:
            logger.debug("No new images to encode.")
    

    def _build_sparse_adjacency_matrix(self):
        """
        构建归一化的稀疏邻接矩阵及其转置矩阵。
        优化：
        1. 严格的缓存检查：同时检查 M 和 M.T 是否存在。
        2. 只在第一次调用时执行耗时的构建和转置操作。
        """
        has_M = hasattr(self, "adj_matrix") and self.adj_matrix is not None
        has_T = hasattr(self, "adj_matrix_T") and self.adj_matrix_T is not None

        if has_M and has_T:
            return self.adj_matrix

        if has_M and not has_T:
            logger.info("Building cached Transposed Adjacency Matrix (M.T)...")
            self.adj_matrix_T = self.adj_matrix.transpose().tocsr()
            return self.adj_matrix

        logger.info("Building sparse adjacency matrix from graph (First time only)...")
        start_t = time.time()
        
        edges = self.graph.get_edgelist()
        weights = self.graph.es["weight"]
        num_nodes = self.graph.vcount()
        
        src, tgt = zip(*edges)
        data = np.array(weights, dtype=np.float32)
        
        if not self.global_config.is_directed_graph:
            src, tgt = list(src) + list(tgt), list(tgt) + list(src)
            data = np.concatenate([data, data])
            
        adj = sp.coo_matrix((data, (src, tgt)), shape=(num_nodes, num_nodes))
        
        row_sum = np.array(adj.sum(1)).flatten()
        row_sum[row_sum == 0] = 1.0 
        d_inv = 1.0 / row_sum
        D_inv = sp.diags(d_inv)
        
        self.adj_matrix = D_inv.dot(adj).tocsr()
        
        logger.info("Caching Transposed Adjacency Matrix (M.T)...")
        self.adj_matrix_T = self.adj_matrix.transpose().tocsr()
        
        logger.info(f"Adjacency matrices built and cached in {time.time()-start_t:.2f}s.")
        return self.adj_matrix

    def run_batch_ppr(self, batch_reset_probs: sp.csr_matrix, target_indices: List[int], max_iter=None, tol=None) -> np.ndarray:
        """
        [GPU Optimized] 极速版 PPR 计算 (支持自动收敛停止)。
        优化点：
        1. 使用预计算的 M.T，避免现场转置。
        2. 循环外预计算 Restart 增量向量。
        3. Pinned Memory 加速数据传输。
        4. [NEW] 基于 L1 范数的收敛检测，提前终止迭代。
        """
        max_iter = max_iter or self.global_config.ppr_max_iter
        tol = tol if tol is not None else self.global_config.ppr_tolerance
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cps
        except ImportError:
            raise ImportError("Please install cupy for GPU acceleration.")

        device_id = getattr(self.global_config, "ppr_device", 0)
        
        self._build_sparse_adjacency_matrix()
        M_T_cpu = self.adj_matrix_T 
        
        num_queries = batch_reset_probs.shape[0]
        BATCH_SIZE = getattr(self.global_config, "ppr_batch_size", 128)

        final_results_list = []

        logger.info(f"Moving PPR computation to GPU {device_id} (Auto-Convergence Mode)...")
        
        with cp.cuda.Device(device_id):
            M_T_gpu = cps.csr_matrix(M_T_cpu)
            
            target_indices_gpu = cp.array(target_indices)
            alpha = self.global_config.ppr_damping
            restart_coeff = 1.0 - alpha

            total_batches = (num_queries + BATCH_SIZE - 1) // BATCH_SIZE
            
            total_iter_count = 0
            
            for i in tqdm(range(0, num_queries, BATCH_SIZE), desc=f"GPU PPR", total=total_batches):
                r_batch_cpu_dense = batch_reset_probs[i : i + BATCH_SIZE].toarray().T
                
                r_batch_gpu_T = cp.array(r_batch_cpu_dense, dtype=cp.float32)
                
                p_batch_gpu_T = r_batch_gpu_T.copy()
                
                r_scaled_gpu_T = r_batch_gpu_T * restart_coeff 
                
                actual_iter = 0
                for iter_step in range(max_iter):
                    actual_iter += 1
                    
                    p_next_T = M_T_gpu.dot(p_batch_gpu_T)
                    p_next_T *= alpha
                    p_next_T += r_scaled_gpu_T
                    
                    diff = p_next_T - p_batch_gpu_T
                    cp.abs(diff, out=diff)
                    
                    max_err = cp.max(cp.sum(diff, axis=0)).item()
                    
                    p_batch_gpu_T = p_next_T
                    
                    if max_err < tol:
                        break
                
                total_iter_count += actual_iter
                
                target_scores_T = p_batch_gpu_T[target_indices_gpu, :]
                batch_scores_cpu = cp.asnumpy(target_scores_T.T)
                final_results_list.append(batch_scores_cpu)
                
                del r_batch_gpu_T, p_batch_gpu_T, p_next_T, r_scaled_gpu_T, target_scores_T, diff
                
                mempool = cp.get_default_memory_pool()
                if mempool.used_bytes() > 0.8 * mempool.total_bytes():
                    mempool.free_all_blocks()

            avg_iter = total_iter_count / total_batches if total_batches > 0 else 0
            logger.info(f"PPR finished. Avg iterations: {avg_iter:.1f}/{max_iter}")

            del M_T_gpu
            cp.get_default_memory_pool().free_all_blocks()

        if not final_results_list:
            return np.array([])
            
        return np.vstack(final_results_list)
    
    
    def qa_infoseek(self, queries: List[QuerySolution], mode: str = "mllm", test_basemodel = False) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        all_qa_messages = []
        MAX_WORKERS = 50
        if mode == "llm":
            for query_solution in queries:
                retrieved_passages = query_solution.docs[: self.global_config.qa_top_k]
                context_user = ""
                question_user = ""
                
                if not test_basemodel:
                    for passage in retrieved_passages:
                        passage = self._truncate_passage(passage=passage, separator="##", is_format=False)
                        context_user += f"Wikipedia Title: {passage}\n\n"
                    
                question_user += query_solution.question
                prompt_template_name = "llm_infoseek_qa"
                all_qa_messages.append(self.prompt_template_manager.render(name=prompt_template_name, context_user=context_user, question_user=question_user))

            logger.info(f"Starting ThreadPoolExecutor with {MAX_WORKERS} workers for LLM inference...")
            infer_with_args = partial(self.llm_model.infer, temperature=self.global_config.temperature, model=self.global_config.llm_name)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                all_qa_results = list(executor.map(infer_with_args, all_qa_messages))
        
        else:
            for query_solution in queries:
                retrieved_passages = query_solution.docs[: self.global_config.qa_top_k]
                context_user = ""
                if not test_basemodel:
                    for passage in retrieved_passages:
                        passage = self._truncate_passage(passage=passage, separator="##", is_format=False)
                        context_user += f"Wikipedia Title: {passage}\n\n"
                
                question_user = query_solution.question
                prompt_template_name = "mllm_infoseek_qa"
                all_qa_messages.append(
                    self.prompt_template_manager.render(name=prompt_template_name, image_input=query_solution.image, context_user=context_user, question_user=question_user)
                )

            logger.info(f"Starting ThreadPoolExecutor with {MAX_WORKERS} workers for MLLM inference...")
            infer_with_args = partial(self.mllm_model.infer, temperature=self.global_config.temperature, model=self.global_config.mllm_name)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                all_qa_results = list(executor.map(infer_with_args, all_qa_messages))

        all_response_message, all_metadata, _ = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)
        queries_solutions = []
        
        for query_solution_idx, query_solution in enumerate(queries):
            query_solution.answer = all_response_message[query_solution_idx]
            queries_solutions.append(query_solution)
        return queries_solutions, all_response_message, all_metadata

    def _truncate_passage(self, passage: str, separator: str = "\n\n", is_format: bool = True) -> str:

        max_len = self.global_config.qa_context_max_len
        token_ids = self.tokenizer.encode(passage, add_special_tokens=False)

        if max_len < 0 or len(token_ids) <= max_len:
            return passage
        
        if not is_format:
            return self.tokenizer.decode(token_ids[:max_len])

        segments = passage.split(separator)
        safe_passage = ""

        for i, segment in enumerate(segments):
            current_sep = separator if i > 0 else ""
            candidate = safe_passage + current_sep + segment
            candidate_ids = self.tokenizer.encode(candidate, add_special_tokens=False)

            if len(candidate_ids) <= max_len:
                safe_passage = candidate
            else:
                if i == 0:
                    return self.tokenizer.decode(token_ids[:max_len])
                break
        return safe_passage
    
