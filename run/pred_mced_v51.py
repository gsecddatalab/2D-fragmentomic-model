import argparse
import os

import pandas as pd
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src import BinaryStacking

MODEL_NAME = "mced_v51"
CONFIG_PATH = "config/mced_stacking_resnet_and_single_task_cnn_v51.yaml"


def _normalize_col_name(col_name: str) -> str:
    return col_name.strip().lower().replace("_", "")


def _find_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_to_original = {
        _normalize_col_name(col_name): col_name for col_name in df.columns
    }
    for candidate in candidates:
        mapped = normalized_to_original.get(_normalize_col_name(candidate))
        if mapped is not None:
            return mapped
    return None


def main():
    parser = argparse.ArgumentParser(description="Run MCED v5.1 prediction")
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        type=str,
        required=True,
        help="Path to the data directory",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to the requested metadata file",
    )

    parser.add_argument(
        "--save_dir", type=str, required=True, help="Save directory for results"
    )

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    config = OmegaConf.load(CONFIG_PATH)
    model = BinaryStacking(config=config, data_dir=args.data_dir)

    if args.metadata.endswith(".csv"):
        df = pd.read_csv(args.metadata)
    else:
        df = pd.read_excel(args.metadata)

    df.drop_duplicates(inplace=True)

    sample_col = _find_first_existing_col(
        df, ["SampleID", "NONBS_ID", "sample_id", "nonbs_id"]
    )
    run_col = _find_first_existing_col(df, ["Runname", "RUN", "Run_NONBS", "run"])
    tsv_col = _find_first_existing_col(df, ["TSV_ID", "tsv_id", "TsvID", "tsv"])

    if sample_col is None or run_col is None:
        raise KeyError(
            "Metadata must include sample and run columns. "
            f"Supported sample columns: SampleID/NONBS_ID. "
            f"Supported run columns: Runname/RUN/Run_NONBS. "
            f"Found columns: {list(df.columns)}"
        )

    # If no explicit tsv column is provided, use sample id as fallback.
    tsv_col = tsv_col or sample_col
    data_samples = df[[sample_col, run_col, tsv_col]].values.tolist()

    sample_ids, run_ids, probs, preds = [], [], [], []

    for sample_id, run_id, tsv_sample in tqdm(data_samples, desc="Processing samples"):
        pred = model(run_id=run_id, sample_id=sample_id, tsv_id=tsv_sample)
        pred = pred.model_dump()

        sample_ids.append(sample_id)
        run_ids.append(run_id)
        probs.append(pred["prob"])
        preds.append(pred["pred_id"])

    res_df = pd.DataFrame(
        {
            "Run": run_ids,
            "SampleID": sample_ids,
            f"{MODEL_NAME}_prob": probs,
            f"{MODEL_NAME}_pred": preds,
        }
    )

    res_df.to_csv(
        f"{args.save_dir}/{os.path.splitext(os.path.basename(args.metadata))[0]}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
