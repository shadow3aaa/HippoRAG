from copy import deepcopy
from typing import Optional
import requests
import json

import numpy as np
from tqdm import tqdm

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


class SiliconflowEmbeddingModel(BaseEmbeddingModel):
    """
    硅基流动嵌入模型适配器
    适配SiliconFlow Embedding API：https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings
    """

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}"
            )

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}"
        )

        # 设置API基础URL和授权
        self.base_url = self.global_config.embedding_base_url or "https://api.siliconflow.cn/v1"
        self.api_key = self.global_config.embedding_api_key

    def _init_embedding_config(self) -> None:
        """
        提取嵌入模型特定参数以初始化EmbeddingConfig。

        Returns:
            None
        """
        # 如果没有指定模型名称，默认使用bge-large-zh-v1.5
        if not hasattr(self, "embedding_model_name") or not self.embedding_model_name:
            self.embedding_model_name = "BAAI/bge-large-zh-v1.5"

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "model": self.embedding_model_name,
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "batch_size": self.global_config.embedding_batch_size,
                "encoding_format": "float",  # 硅基流动API支持的格式
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        对文本列表进行编码，返回嵌入向量数组

        Args:
            texts: 要编码的文本列表

        Returns:
            编码后的嵌入向量数组
        """
        # 清理文本，替换换行符，确保没有空字符串
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != "" else " " for t in texts]

        # 准备请求的URL和头部
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # 准备请求体
        payload = {
            "model": self.embedding_config.model_init_params["model"],
            "input": texts,
            "encoding_format": self.embedding_config.encode_params["encoding_format"],
        }

        try:
            # 发送请求
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # 如果响应包含错误状态码，引发异常

            # 解析响应
            result = response.json()

            # 提取嵌入向量
            embeddings = [item["embedding"] for item in result["data"]]

            # 将结果转换为numpy数组
            return np.array(embeddings)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling SiliconFlow API: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response content: {e.response.content}")
            raise

    def batch_encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        对文本批量编码，处理大批量文本

        Args:
            texts: 要编码的文本列表
            **kwargs: 附加参数，可以包括instruction等

        Returns:
            编码后的嵌入向量数组
        """
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        # 处理instruction，如果提供了指令，将其添加到文本前面
        if "instruction" in kwargs and kwargs["instruction"] != "":
            texts = [f"Instruct: {kwargs['instruction']}\nQuery: {text}" for text in texts]

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")

        batch_size = params.pop("batch_size", 16)

        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                try:
                    results.append(self.encode(batch))
                except Exception as e:
                    logger.error(f"Error in batch encoding: {str(e)}")
                    raise
                pbar.update(len(batch))
            pbar.close()
            results = np.concatenate(results)

        # 如果需要归一化
        if params.get("norm", self.embedding_config.norm):
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
