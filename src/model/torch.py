import numpy as np
import onnxruntime as ort
import torch

from src.data import load_2d_data
from src.schemas import Get2dDataInp

from ._base import ModelEmbeddingAbs


class TorchEmbedding(ModelEmbeddingAbs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _predict(self, sample: np.ndarray):
        if self.config.ckpt_path.endswith(".pt"):
            input_tensor = torch.from_numpy(sample).float().to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)

            return output.detach().cpu().numpy()

        elif self.config.ckpt_path.endswith(".onnx"):
            ort_inputs = {self.model.get_inputs()[0].name: sample.astype(np.float32)}
            ort_outs = self.model.run(None, ort_inputs)
            return ort_outs[0]

    def _load_sample(self, run_id: str, sample_id: str, tsv_id: str) -> np.ndarray:
        inp = Get2dDataInp(
            run_id=run_id,
            sample_id=sample_id,
            tsv_id=tsv_id,
            feature_type=self.config.feature_type,
            num_bins=self.config.num_bins,
            flen_range=list(np.arange(*self.config.flen_range)),
            which_mer=self.config.which_mer,
            feature_increasing_factor=self.config.feature_increasing_factor,
            datalake_path=self.config.data_dir,
        )

        data_sample = load_2d_data(inp)
        data_sample = np.expand_dims(data_sample, axis=0)
        data_sample = np.expand_dims(data_sample, axis=0)

        return data_sample

    def _load_model(self):
        if self.config.ckpt_path.endswith(".pt"):
            feature_extractor = torch.jit.load(
                self.config.ckpt_path, map_location=self.device
            )
            feature_extractor.eval()
        elif self.config.ckpt_path.endswith(".onnx"):
            feature_extractor = ort.InferenceSession(
                self.config.ckpt_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

        return feature_extractor
