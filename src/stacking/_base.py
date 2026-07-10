import os
from typing import Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.model import KerasEmbedding, OneDimEmbedding, TorchEmbedding
from src.schemas import EmbeddingConfig, ModelPrediction


class BaseStacking:
    def __init__(self, config: DictConfig, data_dir: str):
        self.config = config
        self.data_dir = data_dir

        self._validate_paths()
        self._load_stack_model(model_path=self.config.stack_model_path)
        self.embed_models, self.embed_model_names = self._load_embed_models(
            self.config.ckpts
        )

    def _validate_paths(self):
        if not os.path.exists(self.config.stack_model_path):
            raise FileNotFoundError(
                f"Stack model not found: {self.config.stack_model_path}"
            )

        for _, embed_config in self.config.ckpts.items():
            if hasattr(embed_config, "ckpt_path"):
                if not os.path.exists(embed_config.ckpt_path):
                    raise FileNotFoundError(
                        f"Embedding model not found: {embed_config.ckpt_path}"
                    )

    def __call__(
        self, run_id: str, sample_id: str, tsv_id: str = None
    ) -> ModelPrediction:
        features = self._get_embeddings(run_id, sample_id, tsv_id)
        prob, pred_id = self._predict(features)
        return ModelPrediction(prob=prob, pred_id=pred_id)

    def _get_embeddings(self, run_id: str, sample_id: str, tsv_id: str) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement _get_embeddings()")

    def _predict(self, features: np.ndarray) -> Tuple[float, int]:
        """Return probability and prediction ID"""
        raise NotImplementedError("Subclasses must implement _predict()")

    def _load_stack_model(self, model_path: str):
        raise NotImplementedError("Subclasses must implement _load_stack_model()")

    def _load_embed_models(self, ckpt_config: DictConfig) -> list:
        embed_models = list()
        embed_model_names = list()

        for embed_name, embed_config in ckpt_config.items():
            config = EmbeddingConfig(
                ckpt_path=OmegaConf.select(embed_config, "ckpt_path"),
                data_dir=self.data_dir,
                feature_type=embed_config.feature_type,
                which_mer=OmegaConf.select(embed_config, "which_mer") or 4,
                num_bins=OmegaConf.select(embed_config, "num_bins") or 60,
                flen_range=OmegaConf.select(embed_config, "flen_range") or [70, 281, 1],
                feature_increasing_factor=OmegaConf.select(
                    embed_config, "feature_increasing_factor"
                ),
                selected_columns=OmegaConf.select(embed_config, "selected_columns"),
            )

            if embed_name.startswith("singletask_cnn"):
                model_cls = KerasEmbedding
            elif embed_name.startswith("resnet"):
                model_cls = TorchEmbedding
            elif embed_name.startswith("one_dim"):
                model_cls = OneDimEmbedding

            model = model_cls(config=config)

            embed_models.append(model)
            embed_model_names.append(embed_name)

        return embed_models, embed_model_names
