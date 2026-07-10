import numpy as np
import pandas as pd


def downgrade_matrix(matrix, k: int):
    """
    Function to downscale 4-mers feature into smaller ones
    """
    motif_order_kth = -1

    if k == 3:
        motif_order_3mers = pd.read_csv("config/motif/motif_order_3mers.csv")
        motif_order_kth = motif_order_3mers
    elif k == 2:
        motif_order_2mers = pd.read_csv("config/motif/motif_order_2mers.csv")
        motif_order_kth = motif_order_2mers
    elif k == 1:
        motif_order_1mer = pd.read_csv("config/motif/motif_order_1mer.csv")
        motif_order_kth = motif_order_1mer

    assert isinstance(motif_order_kth, pd.DataFrame)
    motif_order_kth = motif_order_kth["motif_order"].values

    em_4mer_ids = matrix.columns

    em_3mer_ids = [id[1:4] for id in em_4mer_ids]
    em_2mer_ids = [id[2:4] for id in em_4mer_ids]
    em_1mer_ids = [id[3:4] for id in em_4mer_ids]

    em_tbl = pd.DataFrame(
        {
            "four_mer": em_4mer_ids,
            "three_mer": em_3mer_ids,
            "two_mer": em_2mer_ids,
            "one_mer": em_1mer_ids,
        }
    )

    em_3mer_ids = list(set(em_3mer_ids))
    em_2mer_ids = list(set(em_2mer_ids))
    em_1mer_ids = list(set(em_1mer_ids))

    if k == 3:
        id = "three_mer"
        em_ids = em_3mer_ids
    elif k == 2:
        id = "two_mer"
        em_ids = em_2mer_ids
    elif k == 1:
        id = "one_mer"
        em_ids = em_1mer_ids

    pick = [em_ids.index(x) if x in em_ids else None for x in em_tbl[id]]
    new_matrix = []

    for pos in set(pick):
        p = [i for i, s in enumerate(pick) if s == pos]
        new_matrix.append(matrix.iloc[:, p].sum(axis=1))

    new_matrix = pd.DataFrame(new_matrix).transpose()
    new_matrix.columns = em_ids
    return new_matrix[motif_order_kth]


def downgrade_kxk_matrix(matrix, k):
    motif_order_4mers = pd.read_csv("config/motif/motif_order_4mers.csv")[
        "motif_order"
    ].values
    matrix.index = motif_order_4mers

    matrix = downgrade_matrix(matrix, k)
    motif_order_kth = -1

    if k == 3:
        motif_order_3mers = pd.read_csv("config/motif/motif_order_3mers.csv")
        motif_order_kth = motif_order_3mers
    elif k == 2:
        motif_order_2mers = pd.read_csv("config/motif/motif_order_2mers.csv")
        motif_order_kth = motif_order_2mers
    elif k == 1:
        motif_order_1mer = pd.read_csv("config/motif/motif_order_1mer.csv")
        motif_order_kth = motif_order_1mer

    assert isinstance(motif_order_kth, pd.DataFrame)

    motif_order_kth = motif_order_kth["motif_order"].values

    em_4mer_ids = matrix.index

    em_3mer_ids = [id[1:4] for id in em_4mer_ids]
    em_2mer_ids = [id[2:4] for id in em_4mer_ids]
    em_1mer_ids = [id[3:4] for id in em_4mer_ids]

    em_tbl = pd.DataFrame(
        {
            "four_mer": em_4mer_ids,
            "three_mer": em_3mer_ids,
            "two_mer": em_2mer_ids,
            "one_mer": em_1mer_ids,
        }
    )

    em_3mer_ids = list(set(em_3mer_ids))
    em_2mer_ids = list(set(em_2mer_ids))
    em_1mer_ids = list(set(em_1mer_ids))

    if k == 3:
        id = "three_mer"
        em_ids = em_3mer_ids
    elif k == 2:
        id = "two_mer"
        em_ids = em_2mer_ids
    elif k == 1:
        id = "one_mer"
        em_ids = em_1mer_ids

    pick = [em_ids.index(x) if x in em_ids else None for x in em_tbl[id]]

    new_matrix = []
    for pos in set(pick):
        p = [i for i, s in enumerate(pick) if s == pos]
        new_matrix.append(matrix.iloc[p, :].sum(axis=0))

    new_matrix = pd.DataFrame(new_matrix).transpose()
    new_matrix.index = em_ids
    new_matrix.columns = motif_order_kth
    return new_matrix.loc[motif_order_kth]


def metadata_to_datapool(
    sample_path: str,
    feature_type: str,
    flen_range: list,
    which_mer: int = 4,
    num_bins: int = 60,
    feature_increasing_factor: int = 1,
):
    assert which_mer in [1, 2, 3, 4]

    feature = pd.read_csv(sample_path)

    if feature_type == "EM_flen":
        if which_mer < 4:
            feature = downgrade_matrix(feature, k=which_mer)
        feature.index = feature.index + 70
        feature_transformed = feature.loc[flen_range, :]

    elif feature_type == "nucleosome_forward_flen":
        feature.index = feature.index + 70
        feature_transformed = []
        for j in range(num_bins):
            feature_transformed.append(
                feature.iloc[
                    :,
                    j * (int(600 / num_bins)) : (j + 1) * (int(600 / num_bins)),
                ].sum(axis=1)
            )
        feature_transformed = pd.concat(feature_transformed, axis=1)

    elif feature_type == "nucleosome_reverse_flen":
        feature.index = feature.index + 70
        feature_transformed = []
        for j in range(num_bins):
            feature_transformed.append(
                feature.iloc[
                    :,
                    j * (int(600 / num_bins)) : (j + 1) * (int(600 / num_bins)),
                ].sum(axis=1)
            )
        feature_transformed = pd.concat(feature_transformed, axis=1)

    elif feature_type == "motif_pairs_all_flen":
        if which_mer < 4:
            feature_transformed = downgrade_kxk_matrix(feature, k=which_mer)
        feature_transformed = feature

    elif feature_type == "EM_forward_nucleosome":
        if which_mer < 4:
            feature = downgrade_matrix(feature, k=which_mer)
        feature_transformed = []
        for j in range(num_bins):
            feature_transformed.append(
                feature.iloc[
                    j * (int(600 / num_bins)) : (j + 1) * (int(600 / num_bins)),
                    :,
                ].sum(axis=0)
            )
        feature_transformed = pd.concat(feature_transformed, axis=1).T

    elif feature_type == "EM_reverse_nucleosome":
        if which_mer < 4:
            feature = downgrade_matrix(feature, k=which_mer)
        feature_transformed = []
        for j in range(num_bins):
            feature_transformed.append(
                feature.iloc[
                    j * (int(600 / num_bins)) : (j + 1) * (int(600 / num_bins)),
                    :,
                ].sum(axis=0)
            )
        feature_transformed = pd.concat(feature_transformed, axis=1).T

    elif (
        feature_type
        == "allEM_forwardNDRd"
        | "allEM_reverseNDRd"
        | "allEM_forwardNDRn"
        | "allEM_reverseNDRn"
        | "allEM_forwardNDRr"
        | "allEM_reverseNDRr"
    ):
        feature_transformed = []
        for j in range(num_bins):
            feature_transformed.append(
                feature.iloc[
                    j * (int(600 / num_bins)) : (j + 1) * (int(600 / num_bins)),
                    :,
                ].sum(axis=0)
            )
        feature_transformed = pd.concat(feature_transformed, axis=1).T

    elif (
        feature_type
        == "forwardNDRd_FLEN"
        | "reverseNDRd_FLEN"
        | "forwardNDRn_FLEN"
        | "reverseNDRn_FLEN"
        | "forwardNDRr_FLEN"
        | "reverseNDRr_FLEN"
    ):

        feature_transformed = []

        for j in range(num_bins):
            feature_transformed.append(
                feature.iloc[
                    :,
                    j * (int(600 / num_bins)) : (j + 1) * (int(600 / num_bins)),
                ].sum(axis=1)
            )
        feature_transformed = pd.concat(feature_transformed, axis=1)

    elif feature_type == "CNAbin_EM":
        feature_transformed = []
        for j in range(num_bins):
            feature_transformed.append(
                feature.iloc[
                    j * (int(2700 / num_bins)) : (j + 1) * (int(2700 / num_bins)),
                    :,
                ].sum(axis=0)
            )
        feature_transformed = pd.concat(feature_transformed, axis=1).T
    elif feature_type == "CNAbin_FLEN":
        feature_transformed = []
        for j in range(num_bins):
            feature_transformed.append(
                feature.iloc[
                    :,
                    j * (int(2700 / num_bins)) : (j + 1) * (int(2700 / num_bins)),
                ].sum(axis=1)
            )
        feature_transformed = pd.concat(feature_transformed, axis=1)

    else:
        raise ValueError(f"[Error] Not catch for {feature_type}")

    feature_transformed = feature_transformed.values
    feature_transformed = feature_transformed * feature_increasing_factor

    return feature_transformed


def metadata_to_datapool_1D(
    feature: pd.DataFrame,
    feature_type: str,
    which_mer: int = 4,
    feature_increasing_factor: int = 1,
    selected_columns=None,
):

    assert which_mer in [1, 2, 3, 4]

    sample_features = []

    if feature_type == "GWfeature_EM":
        feature = feature["freq"]

    elif feature_type == "GWfeature_flen":
        feature = feature[(feature["size"] >= 70) & (feature["size"] <= 280)]["freq"]

    elif feature_type == "GWfeature_Nucleosome":
        feature = feature[(feature["dist"] >= -300) & (feature["dist"] <= 300)]["freq"]

    if selected_columns:
        feature = feature[selected_columns]

    feature = feature.values[0]

    sample_features = np.array(feature)

    return sample_features
