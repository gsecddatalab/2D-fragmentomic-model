from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Get2dDataInp(BaseModel):
    run_id: Optional[str] = None
    sample_id: Optional[str] = None
    tsv_id: Optional[str] = None
    feature_type: str
    datalake_path: str
    num_bins: int
    flen_range: List
    which_mer: Literal[1, 2, 3, 4]
    feature_increasing_factor: int


class Get1dDataInp(BaseModel):
    run_id: Optional[str] = None
    sample_id: Optional[str] = None
    tsv_id: Optional[str] = None
    feature_type: str
    datalake_path: str
    bs_datalake_path: Optional[str] = None
    which_mer: Literal[1, 2, 3, 4] = 4
    feature_increasing_factor: int = 1e4
    selected_columns: Optional[List[str]] = Field(default=None)


class EmbeddingConfig(BaseModel):
    ckpt_path: Optional[str] = None
    data_dir: str
    feature_type: str
    which_mer: int
    num_bins: int = 60
    flen_range: List[int] = [70, 281, 1]
    feature_increasing_factor: int
    selected_columns: Optional[List[str]] = Field(default=None)
    bs_datalake_path: Optional[str] = Field(default=None)


class ModelPrediction(BaseModel):
    prob: float | list[float]
    pred_id: int
