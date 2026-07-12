from .base import EmbeddingConfig, BaseEmbeddingModel
from .eva_clip import EvaClipModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str):
    if "EVA-CLIP" in embedding_model_name:
        return EvaClipModel
    raise ValueError(f"Unsupported embedding model: {embedding_model_name}. MG²-RAG uses EVA-CLIP.")
