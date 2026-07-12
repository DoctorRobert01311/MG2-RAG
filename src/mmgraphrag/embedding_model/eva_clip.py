import os
import torch
import numpy as np
from typing import List, Optional, Union, Any, Dict
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizer
from concurrent.futures import ThreadPoolExecutor

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from ..utils.misc_utils import monitor_vram
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

class EvaClipModel(BaseEmbeddingModel):

    _loaded_cache = {}

    def __init__(
        self,
        global_config: Optional[BaseConfig] = None,
        multimodal_embedding_model_name: Optional[str] = None,
    ) -> None:
        super().__init__(global_config=global_config)

        if multimodal_embedding_model_name is not None:
            self.multimodal_embedding_model_name = multimodal_embedding_model_name

        if not os.path.exists(self.multimodal_embedding_model_name):
            raise FileNotFoundError(f"Path not found: {self.multimodal_embedding_model_name}")
        self._init_embedding_config()
        self.device = "cuda:" + str(self.global_config.clip_device)

        logger.info(f"[eva-clip] Loade from: {self.multimodal_embedding_model_name}, device {self.device}, batchsize {self.global_config.multimodal_embedding_batch_size}")

        cache_key = self.multimodal_embedding_model_name + "_" + self.device
        if cache_key in EvaClipModel._loaded_cache:
            logger.debug(f"Using cached EVA-CLIP model for key: {cache_key}")
            cached_data = EvaClipModel._loaded_cache[cache_key]
            self.tokenizer = cached_data["tokenizer"]
            self.processor = cached_data["processor"]
            self.model = cached_data["model"]
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.multimodal_embedding_model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.processor = CLIPImageProcessor.from_pretrained(self.global_config.clip_image_processor_name, trust_remote_code=True)
            logger.info(f"[clip processor] Loading CLIPImageProcessor from pretrained config {self.global_config.clip_image_processor_name}.")

            with monitor_vram(self.device, task_name="EVA-CLIP Model"):
                self.model = (
                    AutoModel.from_pretrained(
                        self.multimodal_embedding_model_name,
                        torch_dtype=(torch.float16 if self.device.startswith("cuda") else torch.float32),
                        trust_remote_code=True,
                    )
                    .to(self.device)
                    .eval()
                )

            EvaClipModel._loaded_cache[cache_key] = {
                "tokenizer": self.tokenizer,
                "processor": self.processor,
                "model": self.model,
            }

        self.embedding_dim = self.model.config.projection_dim
        logger.debug(f"Found projection_dim in model.config: {self.embedding_dim}")

    def _init_embedding_config(self) -> None:
        config_dict = {
            "multimodal_embedding_model_name": self.multimodal_embedding_model_name,
            "norm": True,
            "encode_params": {
                "batch_size": self.global_config.multimodal_embedding_batch_size,
                "max_length": self.global_config.multimodal_embedding_max_model_len,
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)

    def _encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        def clean_text_for_clip(text: str) -> str:
            if not text:
                return ""
            text = text.replace('\n', ' ').replace('\t', ' ')
            text = " ".join(text.split())
            return text
        if isinstance(texts[0], list): texts = [clean_text_for_clip(t) for t in texts]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.embedding_config.encode_params["max_length"],
        )
        input_ids = inputs.input_ids.to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            text_features = self.model.encode_text(input_ids)
        return text_features

    def _load_single_image(self, img: Union[str, Image.Image]) -> Image.Image:
        """
        辅助函数：用于线程池中加载单张图片。
        有问题返回纯黑函数像素。
        """
        try:
            if isinstance(img, str):
                return Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                return img.convert("RGB")
            else:
                logger.error(f"Unsupported image type: {type(img)}")
                return Image.new("RGB", (224, 224))
        except Exception as e:
            logger.error(f"Error loading image {img}: {e}")
            return Image.new("RGB", (224, 224))

    def _encode_image_batch(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        num_workers = 8 
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            processed_images = list(executor.map(self._load_single_image, images))
        inputs = self.processor(images=processed_images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        if self.model.dtype == torch.float16:
            pixel_values = pixel_values.half()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            image_features = self.model.encode_image(pixel_values)
        return image_features

    def batch_encode(self, texts: List[str] = None, images: List[Union[str, Image.Image]] = None, **kwargs) -> np.ndarray:

        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided.")

        data_source = texts if texts is not None else images
        is_text_mode = texts is not None

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)
        batch_size = params.get("batch_size", 16)
        if is_text_mode:
            batch_size = batch_size * 8
        mode_name = "Text" if is_text_mode else "Image"
        results_list = []

        pbar = tqdm(total=len(data_source), desc=f"EVA-CLIP {mode_name} Encode", unit="sample")
        for i in range(0, len(data_source), batch_size):
            batch_data = data_source[i : i + batch_size]
            if is_text_mode:
                batch_emb = self._encode_text_batch(batch_data)
            else:
                batch_emb = self._encode_image_batch(batch_data)
            results_list.append(batch_emb.cpu())
            pbar.update(len(batch_data))
        pbar.close()
        if len(results_list) == 0:
            return np.array([])
        all_embeddings = torch.cat(results_list, dim=0) 
        if self.embedding_config.norm:
            all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)
        return all_embeddings.float().numpy()
