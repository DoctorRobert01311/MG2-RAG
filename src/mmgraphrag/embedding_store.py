import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal

from copy import deepcopy
import pandas as pd
from PIL import Image
from .utils.misc_utils import compute_mdhash_id, compute_image_hash
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
        - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.debug(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(db_filename, f"vdb_{self.namespace}.parquet")
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {
                "content": text
            }

        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return {}

        existing = self.hash_id_to_row.keys()

        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {
            h: {"hash_id": h, "content": t}
            for h, t in zip(missing_ids, texts_to_encode)
        }

    def insert_strings(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {
                "content": text
            }

        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return

        existing = self.hash_id_to_row.keys()

        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.debug(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist."
        )

        if not missing_ids:
            return {}

        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        missing_embeddings = self.embedding_model.batch_encode(texts = texts_to_encode)
        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = (
                df["hash_id"].values.tolist(),
                df["content"].values.tolist(),
                df["embedding"].values.tolist(),
            )
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {
                h: self.texts[idx] for idx, h in enumerate(self.hash_ids)
            }
            self.text_to_hash_id = {
                self.texts[idx]: h for idx, h in enumerate(self.hash_ids)
            }
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.debug(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        data_to_save = pd.DataFrame(
            {
                "hash_id": self.hash_ids,
                "content": self.texts,
                "embedding": self.embeddings,
            }
        )
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t}
            for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)
        }
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {
            h: self.texts[idx] for idx, h in enumerate(self.hash_ids)
        }
        self.text_to_hash_id = {
            self.texts[idx]: h for idx, h in enumerate(self.hash_ids)
        }
        logger.debug(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        logger.debug(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)

        logger.debug(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id: self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row["content"] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        item = self.embeddings[self.hash_id_to_idx[hash_id]]
        return np.array(item, dtype=dtype)

    def get_embeddings(self, hash_ids, dtype=np.float32) -> np.ndarray:
        if not hash_ids:
            dim = len(self.embeddings[0]) if self.embeddings else 0
            return np.empty((0, dim), dtype=dtype)

        indices = [self.hash_id_to_idx[h] for h in hash_ids]
        
        batch_data = [self.embeddings[i] for i in indices]

        return np.ascontiguousarray(batch_data, dtype=dtype)


class MultimodalEmbeddingStore(EmbeddingStore):
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        super().__init__(embedding_model, db_filename, batch_size, namespace)
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def _is_valid_image_input(self, item: Any) -> bool:
        """检查是否为有效的图片输入（路径字符串 或 PIL Image对象）"""
        if isinstance(item, Image.Image):
            return True
        if isinstance(item, str):
            _, ext = os.path.splitext(item)
            if ext.lower() in self.image_extensions and os.path.exists(item):
                return True
        return False

    def get_hash_id(self, item: Union[str, Image.Image]) -> str:
        """
        获取 Hash ID 的增强版：
        1. 先尝试从内存字典查 (最快)。
        2. 查不到？如果是文件路径或图片对象，现场计算 Hash (兜底)。
        """
        if isinstance(item, str) and item in self.text_to_hash_id:
            return self.text_to_hash_id[item]

        try:
            computed_hash = compute_image_hash(item, prefix=self.namespace + "-")
            if isinstance(item, str):
                self.text_to_hash_id[item] = computed_hash
            return computed_hash
            
        except Exception as e:
            logger.error(f"Could not retrieve or compute hash id for {item}: {e}")
            raise KeyError(f"Hash ID not found and could not be computed for: {item}")
        
    def insert_inputs(self, inputs: List[Union[str, Image.Image]]):
        """
        分流处理：文本交给父类，图片（路径或对象）交给 _insert_images
        """
        image_inputs = []
        text_inputs = []

        for item in inputs:
            if self._is_valid_image_input(item):
                image_inputs.append(item)
            else:
                text_inputs.append(item)

        if text_inputs:
            logger.debug(f"Delegating {len(text_inputs)} text records to parent class.")
            super().insert_strings(text_inputs)

        if image_inputs:
            self._insert_images(image_inputs)

    def _insert_images(self, image_inputs: List[Union[str, Image.Image]]):
        nodes_dict = {}

        for item in image_inputs:
            try:
                img_hash = compute_image_hash(item, prefix=self.namespace + "-")
                nodes_dict[img_hash] = {"content": item}
            except Exception as e:
                info = item if isinstance(item, str) else "<PIL.Image Object>"
                logger.error(f"Failed to hash image input {info}: {e}")

        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return

        existing = self.hash_id_to_row.keys()
        missing_ids = [hid for hid in all_hash_ids if hid not in existing]

        logger.debug(f"Inserting {len(missing_ids)} new image records.")
        if not missing_ids:
            return

        encode_inputs = []
        storage_inputs = []
        valid_ids = []

        for hid in missing_ids:
            raw_content = nodes_dict[hid]["content"]
            
            try:
                if isinstance(raw_content, str):
                    img_obj = Image.open(raw_content).convert("RGB")
                else:
                    img_obj = raw_content.convert("RGB")

                encode_inputs.append(img_obj)

                if isinstance(raw_content, Image.Image):
                    storage_inputs.append("[IMAGE Object]") 
                else:
                    storage_inputs.append(raw_content)
                
                valid_ids.append(hid)

            except Exception as e:
                logger.error(f"Skipping corrupt or unsupported image: {raw_content} | Error: {e}")

        if not valid_ids:
            return

        try:
            embeddings = self.embedding_model.batch_encode(images=encode_inputs)
            
            self._upsert(valid_ids, storage_inputs, embeddings)

        except Exception as e:
            logger.error(f"Failed to encode/save images: {e}")