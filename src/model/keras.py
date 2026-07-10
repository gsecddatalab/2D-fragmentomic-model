import keras
import numpy as np
import onnxruntime as ort
import tensorflow as tf

from src.data import load_2d_data
from src.schemas import Get2dDataInp

from ._base import ModelEmbeddingAbs


class KerasEmbedding(ModelEmbeddingAbs):
    def __init__(self, config):
        super().__init__(config)
        self._force_cpu_predict = False

    def _predict(self, sample: np.ndarray):
        if self.config.ckpt_path.endswith(".keras"):
            # CuDNN mismatches on some hosts can break GPU Conv2D at runtime.
            # Fallback to CPU prediction so inference can continue.
            if self._force_cpu_predict:
                with tf.device("/CPU:0"):
                    return self.model(sample)

            try:
                return self.model(sample)
            except tf.errors.InvalidArgumentError as err:
                err_msg = str(err).lower()
                if "no dnn in stream executor" in err_msg or "cudnn" in err_msg:
                    self._force_cpu_predict = True
                    with tf.device("/CPU:0"):
                        return self.model(sample)
                raise
        elif self.config.ckpt_path.endswith(".onnx"):
            sample = sample.astype("float32")
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name

            embeddings = self.model.run([output_name], {input_name: sample})
            embeddings = embeddings[0]
            return embeddings

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
        data_sample = np.expand_dims(data_sample, axis=-1)
        data_sample = np.expand_dims(data_sample, axis=0)

        return data_sample

    def _load_model(self):
        if self.config.ckpt_path.endswith(".keras"):
            model = keras.saving.load_model(self.config.ckpt_path)

            feature_extractor = tf.keras.Model(
                inputs=model.inputs, outputs=model.layers[-2].output
            )
        elif self.config.ckpt_path.endswith(".onnx"):
            feature_extractor = ort.InferenceSession(
                self.config.ckpt_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

        return feature_extractor
