import glob
import os

import numpy as np
import pandas as pd

from src.schemas import Get1dDataInp, Get2dDataInp

from .utils import metadata_to_datapool, metadata_to_datapool_1D

CNA_short_path = "/module/data/CNA_short.csv"

cnashort_feature = None

if os.path.exists(CNA_short_path):
    cnashort_feature = pd.read_csv(CNA_short_path)


def load_2d_data(inp: Get2dDataInp) -> np.ndarray:
    features = {
        "EM-FLEN": "EM_FLEN",
        "EM-fNUC": "EM_forwardNUC",
        "EM-rNUC": "EM_reverseNUC",
        "MPAF": "EM_all_fragments",
        "MPLF": "EM_long_fragments",
        "MPSF": "EM_short_fragments",
        "fNUC-FLEN": "forwardNUC_FLEN",
        "rNUC-FLEN": "reverseNUC_FLEN",
        # Experimenting features
        "CNAbin-EM": "CNAbin_EM",
        "CNAbin-FLEN": "CNAbin_FLEN",
    }

    cmc_features = {
        "EM-FLEN": "EM_FLEN",
        "EM-fNUC": "allEM_forwardNUC",
        "EM-rNUC": "reverseEM_reverseNUC",
        "MPAF": "EM_all_fragments",
        "MPLF": "EM_long_fragments",
        "MPSF": "EM_short_fragments",
        "fNUC-FLEN": "forwardNUC_FLEN",
        "rNUC-FLEN": "reverseNUC_FLEN",
        # Experimenting features
        "CNAbin-EM": "CNAbin_EM",
        "CNAbin-FLEN": "CNAbin_FLEN",
    }

    old_features = {
        "EM-FLEN": "EM_flen",
        "EM-fNUC": "EM_forward_nucleosome",
        "EM-rNUC": "EM_reverse_nucleosome",
        "MPAF": "motif_pairs_all_flen",
        "MPLF": "motif_pairs_long_flen",
        "MPSF": "motif_pairs_short_flen",
        "fNUC-FLEN": "nucleosome_forward_flen",
        "rNUC-FLEN": "nucleosome_reverse_flen",
        # Experimenting features
        "CNAbin-EM": "CNAbin_EM",
        "CNAbin-FLEN": "CNAbin_FLEN",
    }

    feature_type = inp.feature_type
    sample_id = inp.sample_id
    run_id = inp.run_id
    datalake_path = inp.datalake_path
    tsv_sample_id = inp.tsv_id
    num_bins = inp.num_bins
    flen_range = inp.flen_range
    which_mer = inp.which_mer
    feature_increasing_factor = inp.feature_increasing_factor

    matrix_feature_path = list()
    attempted_paths = []

    def _collect_feature_candidates(base_run_dir: str, feature_name: str) -> list[str]:
        # Support both flat and nested layouts, e.g.:
        # {run}/{prefix-sample}_features/{prefix-sample}_{feature}.csv
        patterns = [
            f"{base_run_dir}/*{sample_id}*_{feature_name}.csv",
            f"{base_run_dir}/**/*{sample_id}*_{feature_name}.csv",
        ]
        matches = []
        for pattern in patterns:
            attempted_paths.append(pattern)
            matches.extend(glob.glob(pattern, recursive=True))
        return sorted(set(matches))

    tsv_feature_path = (
        f"{datalake_path}/{tsv_sample_id}_features/{tsv_sample_id}_{features[feature_type]}.csv"
    )
    attempted_paths.append(tsv_feature_path)
    if os.path.exists(tsv_feature_path):
        matrix_feature_path.append(tsv_feature_path)

    else:
        tsv_cmc_feature_path = (
            f"{datalake_path}/{tsv_sample_id}_features/{tsv_sample_id}_{cmc_features[feature_type]}.csv"
        )
        attempted_paths.append(tsv_cmc_feature_path)
        if os.path.exists(tsv_cmc_feature_path):
            matrix_feature_path.append(tsv_cmc_feature_path)

    if not matrix_feature_path:
        direct_run_cmc_path = (
            f"{datalake_path}/{run_id}/{sample_id}_{cmc_features[feature_type]}.csv"
        )
        attempted_paths.append(direct_run_cmc_path)
        if os.path.exists(direct_run_cmc_path):
            matrix_feature_path.append(direct_run_cmc_path)

    if not matrix_feature_path:
        direct_run_feature_path = (
            f"{datalake_path}/{run_id}/{sample_id}_{features[feature_type]}.csv"
        )
        attempted_paths.append(direct_run_feature_path)
        if os.path.exists(direct_run_feature_path):
            matrix_feature_path.append(direct_run_feature_path)

    # NonBS commercial data may include a prefixed sample id in filename,
    # e.g. 147-KSGAAD33_S95102-S97102_EM_FLEN.csv.
    if not matrix_feature_path and os.path.isdir(f"{datalake_path}/{run_id}"):
        run_dir = f"{datalake_path}/{run_id}"
        matched_paths = _collect_feature_candidates(run_dir, features[feature_type])
        if not matched_paths:
            matched_paths = _collect_feature_candidates(
                run_dir, cmc_features[feature_type]
            )

        if matched_paths:
            matrix_feature_path.append(matched_paths[0])

    if not matrix_feature_path:
        raise FileNotFoundError(
            f"Cannot find feature file for run_id={run_id}, sample_id={sample_id}, "
            f"tsv_id={tsv_sample_id}, feature={features[feature_type]}, "
            f"datalake_path={datalake_path}. Tried: {attempted_paths}"
        )

    samples = metadata_to_datapool(
        sample_path=matrix_feature_path[0],
        feature_type=old_features[feature_type],
        num_bins=num_bins,
        flen_range=flen_range,
        which_mer=which_mer,
        feature_increasing_factor=feature_increasing_factor,
    )
    samples = np.array(samples)
    return samples


def load_1D_data(inp: Get1dDataInp):
    feature_type = inp.feature_type
    datalake_path = inp.datalake_path
    bs_datalake_path = inp.bs_datalake_path
    which_mer = inp.which_mer
    tsv_sample_id = inp.tsv_id
    sample_id = inp.sample_id
    run_id = inp.run_id
    feature_increasing_factor = inp.feature_increasing_factor
    selected_columns = inp.selected_columns

    features = {
        "EM": "GWfeature_EM",
        "FLEN": "GWfeature_flen",
        "NUC": "GWfeature_Nucleosome",
        "CNA_short": "CNA_short",
        "TMfeature_EM": "TMfeature_EM",
        "TMfeature_FLEN": "TMfeature_FLEN",
        "TMfeature_CNA": "TMfeature_CNA",
    }

    cmc_features = {
        "EM": "GWfeature_EM",
        "FLEN": "GWfeature_flen",
        "NUC": "GWfeature_Nucleosome",
        "CNA_short": "CNA_short",
        "TMfeature_EM": "TMfeature_EM",
        "TMfeature_FLEN": "TMfeature_FLEN",
        "TMfeature_CNA": "TMfeature_CNA",
    }

    old_features = {
        "EM": "GWfeature_EM",
        "FLEN": "GWfeature_flen",
        "NUC": "GWfeature_Nucleosome",
        "CNA_short": "CNA_short",
        "TMfeature_EM": "TMfeature_EM",
        "TMfeature_FLEN": "TMfeature_FLEN",
        "TMfeature_CNA": "TMfeature_CNA",
    }

    assert feature_type in features.keys()

    attempted_paths = []

    tsv_feature_path = (
        f"{datalake_path}/{tsv_sample_id}_features/{tsv_sample_id}_{features[feature_type]}.csv"
    )
    attempted_paths.append(tsv_feature_path)
    if os.path.exists(tsv_feature_path):
        feature = pd.read_csv(
            tsv_feature_path
        )
    elif os.path.isdir(f"{datalake_path}/{run_id}"):
        wildcard_candidates = [
            f"{datalake_path}/{run_id}/*{sample_id}*_{features[feature_type]}.csv",
            f"{datalake_path}/{run_id}/**/*{sample_id}*_{features[feature_type]}.csv",
            f"{datalake_path}/{run_id}/*{sample_id}*_{cmc_features[feature_type]}.csv",
            f"{datalake_path}/{run_id}/**/*{sample_id}*_{cmc_features[feature_type]}.csv",
        ]
        matched_paths = []
        for candidate in wildcard_candidates:
            attempted_paths.append(candidate)
            matched_paths.extend(glob.glob(candidate, recursive=True))

        if matched_paths:
            feature = pd.read_csv(sorted(set(matched_paths))[0])
        else:
            feature = None
    elif bs_datalake_path and os.path.exists(
        f"{bs_datalake_path}/{features[feature_type]}.csv"
    ):
        feature = pd.read_csv(f"{bs_datalake_path}/{features[feature_type]}.csv")
        if "Unnamed: 0" in feature.columns:
            feature = feature.drop(columns=["Unnamed: 0"])

        feature = load_bs_data(
            feature, sample_id, feature_type, feature_increasing_factor
        )

    elif os.path.exists(f"{datalake_path}/{features[feature_type]}.csv"):
        feature = cnashort_feature[cnashort_feature["SampleID"] == sample_id]

    else:
        raise FileNotFoundError(
            f"{run_id} - {sample_id} - {tsv_sample_id} - {features[feature_type]} at {datalake_path} not found"
        )

    if feature is None:
        raise FileNotFoundError(
            f"{run_id} - {sample_id} - {tsv_sample_id} - {features[feature_type]} "
            f"at {datalake_path} not found. Tried: {attempted_paths}"
        )

    if isinstance(feature, pd.DataFrame):
        samples = metadata_to_datapool_1D(
            feature=feature,
            feature_type=old_features[feature_type],
            which_mer=which_mer,
            feature_increasing_factor=feature_increasing_factor,
            selected_columns=selected_columns,
        )

    else:
        samples = feature

    samples = np.array(samples)
    return samples


def load_bs_data(
    feature: pd.DataFrame,
    sample_id: str,
    feature_type: str,
    feature_increasing_factor: int,
):
    find_feature = feature[feature["SampleID"] == sample_id]
    if find_feature.empty:
        find_feature = feature[feature["SampleID"] == f"0AA{sample_id}"]

    feature = find_feature.copy()

    feature.drop(columns=["SampleID"], inplace=True)

    if "em" in feature_type.lower():
        motif_orders = pd.read_csv("config/motif/motif_order_4mers.csv")
        motif_orders = motif_orders.motif_order
        feature = feature[motif_orders]

    feature = feature.values[0]

    feature = feature * feature_increasing_factor

    return feature
