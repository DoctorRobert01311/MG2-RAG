from .base import GroundingModel
import os
import json
from typing import Union, Dict, List, Any, Iterator, Optional
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.eval.postprocessors import PostProcessImage

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

logger = get_logger(__name__)


class SAM3Grounding(GroundingModel):
    def __init__(
        self,
        global_config: Optional[BaseConfig] = None,
        grounding_model_name: Optional[str] = None,
        global_work_dir: Optional[str] = None,
    ):
        """
        初始化 SAM3 模型并设置工作目录。

        Args:
            grounding_model_name: 模型名称 (checkpoint 路径)
            global_work_dir: 根工作目录
        """
        super().__init__(global_config=global_config)
        if grounding_model_name is not None:
            self.grounding_model_name = grounding_model_name
        
        if global_work_dir is not None:
            self.grounding_work_dir = os.path.join(global_work_dir, self.global_config.grounding_model_work_dir)

        self.device = "cuda:"+str(self.global_config.sam3_device)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            
        self.prompt_batch_size = self.global_config.grounding_model_prompt_batch_size
        self.model = None
        
        self._query_id_counter = 0

        self._init_transforms()
        self._init_postprocessor()
        self._load_model(grounding_model_name=self.grounding_model_name)

    def _init_transforms(self):
        """初始化图像预处理流水线"""
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _init_postprocessor(self):
        """初始化结果后处理器"""
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=self.global_config.grounding_detection_threshold,
            to_cpu=False,
        )

    def _load_model(self, grounding_model_name: str):
        try:
            os.makedirs(self.grounding_work_dir, exist_ok=True)
            logger.info(f"Working directory set to: {self.grounding_work_dir}")
            self.model = build_sam3_image_model(checkpoint_path=grounding_model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"[SAM3] Loade:{grounding_model_name}, cuda:{self.device}, batchsize{self.prompt_batch_size}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 model: {e}")

    def _create_datapoint(self, image_pil: Image.Image) -> Datapoint:
        """创建一个包含图像的 Datapoint 对象"""
        w, h = image_pil.size
        return Datapoint(
            find_queries=[],
            images=[SAMImage(data=image_pil, objects=[], size=[h, w])]
        )

    def _add_text_prompt(self, datapoint: Datapoint, text_query: str) -> int:
        """向 Datapoint 添加文本 Prompt，返回唯一的 query ID"""
        assert len(datapoint.images) == 1
        w, h = datapoint.images[0].size
        
        self._query_id_counter += 1
        query_id = self._query_id_counter

        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[], 
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=query_id,
                    original_image_id=query_id,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
        )
        return query_id

    def predict(self, image: Union[str, Image.Image], prompt: Union[str, List[str]], prompt_batch_size: int = 8) -> Dict:
        """
        运行 SAM3 推理 
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
        else:
            raise ValueError("Input image must be a file path or PIL Image.")

        image_np = np.array(image_pil)
        prompts_list = [prompt] if isinstance(prompt, str) else prompt

        if not prompts_list:
            return self._empty_result(image_np, prompts_list)

        all_masks, all_boxes, all_scores, all_label_indices = [], [], [], []
        total_prompts = len(prompts_list)

        with tqdm(total=total_prompts, desc="SAM3 Inference one img", unit="prompt") as pbar:
            
            for chunk_start in range(0, total_prompts, prompt_batch_size):
                chunk_end = min(chunk_start + prompt_batch_size, total_prompts)
                chunk_prompts = prompts_list[chunk_start:chunk_end]
                
                global_index_offset = chunk_start 

                datapoint = self._create_datapoint(image_pil)
                chunk_query_ids = []
                
                for p_text in chunk_prompts:
                    q_id = self._add_text_prompt(datapoint, p_text)
                    chunk_query_ids.append(q_id)

                datapoint_tf = self.transform(datapoint)
                batch = collate([datapoint_tf], dict_key="dummy")["dummy"]
                batch = copy_data_to_device(batch, self.device, non_blocking=True)

                with torch.inference_mode():
                    output = self.model(batch)
                    processed_results = self.postprocessor.process_results(output, batch.find_metadatas)

                for local_idx, q_id in enumerate(chunk_query_ids):
                    res = processed_results.get(q_id)
                    if res is None:
                        continue
                    
                    masks = res.get("masks")
                    boxes = res.get("boxes")
                    scores = res.get("scores")

                    if masks is not None and len(masks) > 0:
                        if masks.dim() == 3:
                            masks = masks.unsqueeze(1)
                        
                        all_masks.append(masks)
                        all_boxes.append(boxes)
                        all_scores.append(scores)
                        
                        real_idx = global_index_offset + local_idx
                        label_tensor = torch.full(
                            (len(masks),), real_idx, dtype=torch.long, device=self.device
                        )
                        all_label_indices.append(label_tensor)
                

                pbar.update(len(chunk_prompts))

        if not all_masks:
            return self._empty_result(image_np, prompts_list)

        return {
            "masks": torch.cat(all_masks, dim=0),
            "boxes": torch.cat(all_boxes, dim=0),
            "scores": torch.cat(all_scores, dim=0),
            "labels": torch.cat(all_label_indices, dim=0),
            "prompts": prompts_list,
            "image_np": image_np,
        }

    def _empty_result(self, image_np, prompts_list):
        return {
            "masks": torch.empty((0, 1, *image_np.shape[:2]), device=self.device),
            "boxes": torch.empty((0, 4), device=self.device),
            "scores": torch.empty((0,), device=self.device),
            "labels": torch.empty((0,), dtype=torch.long, device=self.device),
            "prompts": prompts_list,
            "image_np": image_np,
        }

    def _process_single_crop(
        self, 
        idx: int, 
        mask_np: np.ndarray, 
        box: np.ndarray, 
        image_np: np.ndarray, 
        label: str, 
        score: float, 
        prompts: list, 
        labels_np: np.ndarray,
        save_dir: str, 
        background_color: int
    ) -> Dict:
        """
        [辅助函数] 单个裁剪任务的原子操作：裁剪 -> 合成 -> 保存
        这个函数会在线程池中运行。
        """
        img_h, img_w = image_np.shape[:2]
        
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        crop_w = x2 - x1
        crop_h = y2 - y1
        MIN_SIZE = 10
        
        if crop_w < MIN_SIZE or crop_h < MIN_SIZE:
            logger.info(f"Skipping crop idx={idx} due to small size: ({crop_w}x{crop_h})")
            return None 

        if x2 <= x1 or y2 <= y1:
            return None

        crop_mask = mask_np[y1:y2, x1:x2]
        crop_image = image_np[y1:y2, x1:x2]

        crop_mask_3ch = crop_mask[:, :, None] 
        
        bg = np.full_like(crop_image, background_color)
        out_crop = crop_image * crop_mask_3ch + bg * (1 - crop_mask_3ch)

        clean_class_name = label.replace(" ", "_").replace("/", "-")
        filename = f"obj_{idx}_{clean_class_name}.png"
        final_save_path = os.path.join(save_dir, filename)

        pil_img = Image.fromarray(out_crop.astype(np.uint8))
        
        pil_img.save(final_save_path, compress_level=1)

        return {
            "local_id": idx,
            "filename": filename,
            "label": label,
            "score": round(float(score), 4),
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "save_path": final_save_path,
            "pil_image": pil_img
        }

    def get_save_crop(
        self,
        results: Dict,
        chunk_id: Union[str, int],
        img_id: Union[str, int],
        background_color: int = 0,
        max_workers: int = 8
    ) -> List[Dict[str, Any]]:
        """
        多线程并发优化版：获取裁剪对象并保存。
        """
        if chunk_id is None or img_id is None:
            raise ValueError("Both 'chunk_id' and 'img_id' are required.")

        save_dir = os.path.join(self.grounding_work_dir, str(chunk_id), str(img_id))
        os.makedirs(save_dir, exist_ok=True)

        image_np = results.get("image_np")
        if image_np is not None:
            img_h, img_w = image_np.shape[:2]
        else:
            img_h, img_w = 0, 0
        
        objects_meta = []
        return_list = []

        masks = results["masks"]
        if masks is not None and len(masks) > 0:
    
            masks_np = (
                masks.detach().cpu().numpy().squeeze(1).astype(np.uint8)
                if isinstance(masks, torch.Tensor) 
                else np.array(masks).squeeze(1).astype(np.uint8)
            )
            
            boxes = results["boxes"]
            boxes_np = (
                boxes.detach().cpu().numpy().astype(int)
                if isinstance(boxes, torch.Tensor)
                else np.array(boxes).astype(int)
            )
            
            scores = results["scores"]
            scores_np = (
                scores.detach().cpu().numpy()
                if isinstance(scores, torch.Tensor)
                else np.array(scores)
            )
            
            labels = results["labels"]
            labels_np = (
                labels.detach().cpu().numpy().astype(int)
                if isinstance(labels, torch.Tensor)
                else np.array(labels).astype(int)
            )
            
            prompts = results["prompts"]
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i in range(len(masks_np)):
                    future = executor.submit(
                        self._process_single_crop,
                        idx=i,
                        mask_np=masks_np[i],
                        box=boxes_np[i],
                        image_np=image_np,
                        label=prompts[labels_np[i]],
                        score=scores_np[i],
                        prompts=prompts,
                        labels_np=labels_np,
                        save_dir=save_dir,
                        background_color=background_color
                    )
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        res = future.result()
                        if res:
                            meta_item = {k: v for k, v in res.items() if k in ["local_id", "filename", "label", "score", "box"]}
                            meta_item["rel_path"] = os.path.join(str(chunk_id), str(img_id), meta_item["filename"])
                            objects_meta.append(meta_item)

                            return_list.append({
                                "image": res["pil_image"],
                                "save_path": res["save_path"],
                                "label": res["label"],
                                "score": res["score"],
                                "box": res["box"]
                            })
                    except Exception as e:
                        logger.error(f"Error processing crop: {e}")

            objects_meta.sort(key=lambda x: x["local_id"])
            return_list.sort(key=lambda x: int(os.path.basename(x["save_path"]).split('_')[1]))

        metadata = {
            "chunk_id": chunk_id,
            "img_id": img_id,
            "total_objects": len(objects_meta),
            "image_size": [img_w, img_h],
            "objects": objects_meta,
        }

        json_path = os.path.join(save_dir, "metadata.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
        if len(objects_meta) == 0:
            logger.debug(f"No objects detected. Saved empty metadata to {save_dir}")
        else:
            logger.debug(f"Saved {len(objects_meta)} objects to {save_dir}")

        return return_list
if __name__ == "__main__":
    CHECKPOINT = "/data1/sijundai/models/facebook/sam3/sam3.pt"
    RAW_IMG_PATH = "/home/dsj/sijundai/image/test.jpg"
    WORK_DIR = "/home/dsj/sijundai/data"
    
    global_config = BaseConfig(
        grounding_model_name=CHECKPOINT,
        grounding_model_work_dir="sam3_results",
    )
    
    sam3_agent = SAM3Grounding(global_config=global_config, global_work_dir=WORK_DIR)

    current_chunk_id = "doc_v1_001"
    current_img_id = "image_059" 

    prompts = ["people", "trees", "canoe"]
    logger.info(f"Processing Chunk: {current_chunk_id} | Img: {current_img_id}")

    results = sam3_agent.predict(image=RAW_IMG_PATH, prompt=prompts)

    save_path = sam3_agent.get_save_crop(
        results=results, chunk_id=current_chunk_id, img_id=current_img_id
    )

    logger.info(f"Result saved at: {save_path}")
