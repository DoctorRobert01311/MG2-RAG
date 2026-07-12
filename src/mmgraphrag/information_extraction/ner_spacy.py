import spacy
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Any
import torch
from tqdm import tqdm
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
logger = get_logger(__name__)


class SpacyNER:
    def __init__(self, global_config: BaseConfig):
        self.global_config = global_config
        self.device = self.global_config.ieNer_device
        self.batch_size = self.global_config.ie_batch_size
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        spacy.require_gpu(self.device)
        model_name = self.global_config.ie_model_name
        self.nlp = spacy.load(model_name)
        if not self.nlp.has_pipe("sentencizer"):
            self.nlp.add_pipe("sentencizer", first=True)
        logger.info(f"[SpacyNER] Loade:{model_name}, cuda:{self.device}, batchsize{self.batch_size}.")

    def batch_ieNer(self, hash_id_to_chunk: dict) -> Dict[str, Dict[str, Any]]:
        """
        批量 NER + Relation Extraction，返回统一结构
        :return: {
            "doc_id": {
                "entities": { "Apple": "ORG", ... },
                "sentences": { "Sentence...": ["Apple"], ... },
                "relations": [ {"head": "Steve", "relation": "founded", "tail": "Apple"}, ... ]
            },
            ...
        }
        """
        data_list = [(text, doc_id) for doc_id, text in hash_id_to_chunk.items()]
        unified_results = {}
        docs_stream = self.nlp.pipe(
            data_list, 
            batch_size=self.batch_size, 
            as_tuples=True,
        )
        
        
        logger.info(f"Starting Spacy pipeline for {len(data_list)} docs...")
        
        for doc, doc_id in tqdm(docs_stream, total=len(data_list), desc="NER Processing"):
            entity_map, sent_map, relations = self._extract_single_doc(doc)
            unified_results[doc_id] = {
                "entities": entity_map,
                "sentences": sent_map,
                "relations": relations
            }
            
        return unified_results
            
    def _extract_single_doc(self, doc) -> tuple[Dict[str, str], Dict[str, List[str]], List[Dict[str, str]]]:
        unique_entities_map = {}
        sentence_to_entities = defaultdict(set)

        for ent in doc.ents:
            if ent.label_ in ["ORDINAL", "CARDINAL"]:
                continue
            ent_text = ent.text
            sent_text = ent.sent.text.strip()
            unique_entities_map[ent_text] = ent.label_
            sentence_to_entities[sent_text].add(ent_text)
        
        triplets = self._extract_relations(doc)
        sent_map_list = {k: list(v) for k, v in sentence_to_entities.items()}
        return unique_entities_map, sent_map_list, triplets

    def _extract_relations(self, doc) -> List[Dict[str, str]]:
        """
        从 doc 中提取三元组关系
        包含被动语态修正、倒装句支持、名词修饰与同位语支持
        """
        triplets = []
        
        IGNORE_LABELS = {"ORDINAL", "CARDINAL", "PERCENT", "QUANTITY"} 
        
        VALID_OBJ_DEPS = {
            "dobj",

            "attr",

            "acomp",

            "dative",

            "oprd",

            "npadvmod"
        }

        for sent in doc.sents:
            valid_ents = [e for e in sent.ents if e.label_ not in IGNORE_LABELS]
            if len(valid_ents) < 2:
                continue
            ent_to_root = {e: e.root for e in valid_ents}
            verb_negation_cache = {}
            def get_negation(verb_token):
                if verb_token not in verb_negation_cache:
                    verb_negation_cache[verb_token] = "not " if any(c.dep_ == "neg" for c in verb_token.children) else ""
                return verb_negation_cache[verb_token]
            for e1, e2 in combinations(valid_ents, 2):
                e1_root = ent_to_root[e1]
                e2_root = ent_to_root[e2]
                head_ent, tail_ent, relation = None, None, None   
                if e1_root.head == e2_root.head:
                    verb = e1_root.head
                    if verb.pos_ in ("VERB", "AUX"):
                        negation = get_negation(verb)
                        if e1_root.dep_ in ("nsubj", "csubj") and e2_root.dep_ in VALID_OBJ_DEPS:
                            head_ent, tail_ent = e1.text, e2.text
                            relation = f"{negation}{verb.lemma_}" 
                        elif e2_root.dep_ in ("nsubj", "csubj") and e1_root.dep_ in VALID_OBJ_DEPS:
                            head_ent, tail_ent = e2.text, e1.text
                            relation = f"{negation}{verb.lemma_}"
                elif e2_root.head.head == e1_root.head: 
                    verb = e1_root.head
                    prep = e2_root.head           
                    if verb.pos_ in ("VERB", "AUX") and prep.pos_ == "ADP":
                        negation = get_negation(verb)
                        if e1_root.dep_ in ("nsubj", "csubj") and e2_root.dep_ == "pobj":
                            head_ent, tail_ent = e1.text, e2.text
                            relation = f"{negation}{verb.lemma_} {prep.text}"
                        elif e1_root.dep_ == "nsubjpass" and e2_root.dep_ == "pobj":
                            if prep.text.lower() == "by":
                                head_ent, tail_ent = e2.text, e1.text
                                relation = f"{negation}{verb.lemma_}"
                elif e1_root.head.head == e2_root.head:
                    verb = e2_root.head
                    prep = e1_root.head
                    if verb.pos_ in ("VERB", "AUX") and prep.pos_ == "ADP":
                        negation = get_negation(verb)
                        if e2_root.dep_ in ("nsubj", "csubj") and e1_root.dep_ == "pobj":
                            head_ent, tail_ent = e2.text, e1.text
                            relation = f"{negation}{verb.lemma_} {prep.text}"
                if not relation:
                    if e1_root.head == e2_root:
                        if e1_root.dep_ in ("poss", "compound"): 
                            head_ent, tail_ent = e2.text, e1.text
                            relation = "has_attribute" 
                        elif e1_root.dep_ == "appos": 
                            head_ent, tail_ent = e2.text, e1.text
                            relation = "is_also"
                    elif e2_root.head == e1_root:
                        if e2_root.dep_ == "appos": 
                            head_ent, tail_ent = e1.text, e2.text
                            relation = "located_in" if e2.label_ == "GPE" else "is_also"
                    elif e2_root.head.head == e1_root:
                        if e2_root.dep_ == "pobj":
                            prep = e2_root.head
                            if prep.text in ("in", "at", "on", "from"):
                                head_ent, tail_ent = e1.text, e2.text
                                relation = "located_in"
                            elif prep.text == "of": 
                                head_ent, tail_ent = e2.text, e1.text
                                relation = "related_to"
                if head_ent and tail_ent and relation:
                    triplets.append({
                        "head": head_ent,
                        "relation": relation,
                        "tail": tail_ent
                    })
        return triplets

if __name__ == "__main__":
    
    MODEL_PATH = "/data3/sijundai/models/en_core_web_trf"
    global_config = BaseConfig(
        ie_model_name=MODEL_PATH,
        iener_device=0,
        ie_batch_size=2
    )
    ner = SpacyNER(global_config=global_config)
    
    data = {
        "chunk_101": "Google was founded by Larry Page. It is based in California.",
        "chunk_102": "Microsoft announced new AI features."
    }
    
    results = ner.batch_iener(data)
    import json
    print("=== Unified Output Structure ===")
    print(json.dumps(results, indent=2))

