import pickle

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.model import KerasEmbedding, OneDimEmbedding, TorchEmbedding
from src.schemas import EmbeddingConfig, ModelPrediction

from ._base import BaseStacking


class BinaryStacking(BaseStacking):
    def _predict(self, features: np.ndarray) -> tuple[float, int]:
        prob = self.stack_model.predict_proba(features)
        prob = [s[1] for s in prob]
        pred_id = [1 if s >= self.cutoff else 0 for s in prob]

        return prob[0], pred_id[0]

    def _load_stack_model(self, model_path: str):
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)

        self.stack_model = loaded["pipeline"]
        self.cutoff = loaded["cutoff"]

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


class BsNonBSStacking(BaseStacking):
    def __init__(self, config: DictConfig, data_dir: str, data_bs_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.data_bs_dir = data_bs_dir

        self._validate_paths()
        self._load_stack_model(model_path=self.config.stack_model_path)
        self.embed_models, self.embed_model_names = self._load_embed_models(
            self.config.ckpts
        )

        self.nonbs_model_ids = []
        self.bs_model_ids = []

        for model_idx, model_name in enumerate(self.embed_model_names):
            if not model_name.startswith("one_dim"):
                self.nonbs_model_ids.append(model_idx)
            else:
                self.bs_model_ids.append(model_idx)

    def _load_embed_models(self, ckpt_config: DictConfig) -> list:
        embed_models = list()
        embed_model_names = list()

        for embed_name, embed_config in ckpt_config.items():
            # Use data_bs_dir for one_dim models, data_dir for others
            if embed_name.startswith("one_dim"):
                data_dir_to_use = self.data_bs_dir
            else:
                data_dir_to_use = self.data_dir

            config = EmbeddingConfig(
                ckpt_path=OmegaConf.select(embed_config, "ckpt_path"),
                data_dir=data_dir_to_use,
                feature_type=embed_config.feature_type,
                which_mer=OmegaConf.select(embed_config, "which_mer") or 4,
                num_bins=OmegaConf.select(embed_config, "num_bins") or 60,
                flen_range=OmegaConf.select(embed_config, "flen_range") or [70, 281, 1],
                feature_increasing_factor=OmegaConf.select(
                    embed_config, "feature_increasing_factor"
                ),
                selected_columns=OmegaConf.select(embed_config, "selected_columns"),
                bs_datalake_path=self.data_bs_dir
                if embed_name.startswith("one_dim")
                else None,
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

    def __call__(
        self,
        nonbs_run_id: str,
        nonbs_sample_id: str,
        bs_run_id: str,
        bs_sample_id: str,
        tsv_id: str = None,
    ) -> ModelPrediction:
        features = self._get_embeddings(
            nonbs_run_id, nonbs_sample_id, bs_run_id, bs_sample_id, tsv_id
        )
        prob, pred_id = self._predict(features)
        return ModelPrediction(prob=prob, pred_id=pred_id)

    def _predict(self, features: np.ndarray) -> tuple[float, int]:
        prob = self.stack_model.predict_proba(features)
        prob = [s[1] for s in prob]
        pred_id = [1 if s >= self.cutoff else 0 for s in prob]

        return prob[0], pred_id[0]

    def _load_stack_model(self, model_path: str):
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)

        if "scaler" in loaded:
            self.scaler = loaded["scaler"]
            self.stack_model = loaded["model"]
            self.cutoff = loaded["cutoff"]
        else:
            pipeline = loaded["pipeline"]
            self.scaler = pipeline.named_steps["scaler"]
            self.model = pipeline.named_steps["classifier"]
            self.cutoff = np.float64(loaded["cutoff"])

    def _get_embeddings(
        self,
        nonbs_run_id: str,
        nonbs_sample_id: str,
        bs_run_id: str,
        bs_sample_id: str,
        tsv_id: str = None,
    ) -> np.ndarray:
        bs_embeddings = []
        nonbs_embeddings = []
        for model_idx, model in enumerate(self.embed_models):
            if model_idx in self.nonbs_model_ids:
                embedding = model(nonbs_run_id, nonbs_sample_id, tsv_id)
            else:
                embedding = model(bs_run_id, bs_sample_id, tsv_id)

            if embedding.ndim != 2:
                raise ValueError(
                    f"Found embedding in shape: {embedding.shape}, expected (batch_size, feature_size)"
                )

            if model_idx in self.nonbs_model_ids:
                nonbs_embeddings.append(embedding)
            elif model_idx in self.bs_model_ids:
                bs_embeddings.append(embedding)

        nonbs_embeddings = np.concatenate(nonbs_embeddings, axis=1)
        bs_embeddings = np.concatenate(bs_embeddings, axis=1)

        nonbs_embeddings = self.scaler.transform(nonbs_embeddings)

        embeddings = [bs_embeddings, nonbs_embeddings]

        return np.concatenate(embeddings, axis=1)
