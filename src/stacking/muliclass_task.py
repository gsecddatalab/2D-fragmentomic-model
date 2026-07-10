import pickle

import numpy as np

from src.schemas import ModelPrediction

from ._base import BaseStacking


class MultiClassStacking(BaseStacking):
    def _predict(self, features: np.ndarray) -> tuple[float, int]:
        prob = self.pipeline.predict_proba(features)
        pred_id = np.argmax(prob, axis=1)
        prob = prob.tolist()[0]

        return prob, pred_id[0]

    def __call__(
        self, run_id: str, sample_id: str, tsv_id: str = None
    ) -> ModelPrediction:
        features = self._get_embeddings(run_id, sample_id, tsv_id)
        prob, pred_id = self._predict(features)
        return ModelPrediction(prob=prob, pred_id=pred_id)

    def _load_stack_model(self, model_path: str):
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)

        self.pipeline = loaded["pipeline"]

    def _get_embeddings(self, run_id: str, sample_id: str, tsv_id: str) -> np.ndarray:
        embeddings = []
        for model in self.embed_models:
            embedding = model(run_id, sample_id, tsv_id)
            if embedding.ndim != 2:
                raise ValueError(
                    f"Found embedding in shape: {embedding.shape}, expected (batch_size, feature_size)"
                )
            embeddings.append(embedding)
        return np.concatenate(embeddings, axis=1)
