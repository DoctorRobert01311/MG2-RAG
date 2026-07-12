

from typing import Union, Dict, List
import numpy as np
from PIL import Image
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
logger = get_logger(__name__)

class GroundingModel:
    """
    Grounding 模型的抽象基类
    定义了加载模型、推理和保存结果的标准接口
    """
    global_config: BaseConfig
    grounding_model_name: str
    grounding_work_dir: str
    device: str

    def __init__(self, global_config: BaseConfig = None):
        if global_config is None:
            logger.debug(
                "global config is not given. Using the default ExperimentConfig instance."
            )
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        logger.debug(
            f"Loading {self.__class__.__name__} with global_config: {str(self.global_config)}"
        )
        self.grounding_model_name = self.global_config.grounding_model_name
        self.work_dir = self.global_config.grounding_model_work_dir

        logger.debug(
            f"Init {self.__class__.__name__}'s grounding_model_name with: {self.grounding_model_name}"
        )


    def _load_model(self, checkpoint_path: str, **kwargs) -> None:
        """加载模型权重"""
        raise NotImplementedError


    def predict(self, image: Union[str, Image.Image],  prompt: Union[str, List[str]]) -> Dict:
        """
        执行推理
        :param image: 图片路径或 PIL Image 对象
        :param prompt: 文本提示
        :return: 包含 masks, boxes, scores 的字典
        """
        raise NotImplementedError
    
    



