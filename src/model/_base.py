import os
from abc import ABC, abstractmethod

import numpy as np

from src.schemas import EmbeddingConfig


class EmbeddingAbs(ABC):
    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @abstractmethod
    def __call__(
        self, run_id: str = None, sample_id: str = None, tsv_id: str = None
    ) -> np.ndarray:
        """Return the feature embeddings"""

    @abstractmethod
    def _load_sample(self, run_id: str, sample_id: str, tsv_id=str) -> np.ndarray:
        """Load and prepare a data sample for prediction."""


class ModelEmbeddingAbs(EmbeddingAbs):
    def __init__(self, config: EmbeddingConfig):
        self.config = config

        if not os.path.exists(self.config.ckpt_path):
            raise FileNotFoundError(f"Checkpoint: {self.config.ckpt_path} not found")

        self.model = self._load_model()

    def __call__(
        self, run_id: str = None, sample_id: str = None, tsv_id: str = None
    ) -> np.ndarray:
        data_sample = self._load_sample(
            run_id=run_id, sample_id=sample_id, tsv_id=tsv_id
        )
        embedding = np.array(self._predict(data_sample))
        return embedding

    def _predict(self, sample: np.ndarray):
        return self.model.predict(sample)

    @abstractmethod
    def _load_model(self) -> any:
        """Load the model and return the feature extractor."""
