import numpy as np

from src.data import load_1D_data
from src.schemas import Get1dDataInp

from ._base import EmbeddingAbs


class OneDimEmbedding(EmbeddingAbs):
    def _load_sample(self, run_id: str, sample_id: str, tsv_id: str) -> np.ndarray:
        inp = Get1dDataInp(
            run_id=run_id,
            sample_id=sample_id,
            tsv_id=tsv_id,
            feature_type=self.config.feature_type,
            bs_datalake_path=self.config.bs_datalake_path,
            which_mer=self.config.which_mer,
            feature_increasing_factor=self.config.feature_increasing_factor,
            datalake_path=self.config.data_dir,
            selected_columns=self.config.selected_columns,
        )

        data_sample = load_1D_data(inp)
        data_sample = np.expand_dims(data_sample, axis=0)

        return data_sample

    def __call__(
        self, run_id: str = None, sample_id: str = None, tsv_id: str = None
    ) -> np.ndarray:
        data_sample = self._load_sample(
            run_id=run_id, sample_id=sample_id, tsv_id=tsv_id
        )

        return data_sample
