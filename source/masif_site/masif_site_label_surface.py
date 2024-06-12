import os
import sys
import importlib
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import pymesh
from default_config.masif_opts import masif_opts

"""
masif_site_label_surface.py: Color a protein ply surface file by the MaSIF-site interface score.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

logger = logging.getLogger(__name__)

os.environ["TF_USE_LEGACY_KERAS"] = "1"
tf.compat.v1.disable_v2_behavior()

params = masif_opts["site"]
custom_params_file = sys.argv[1]
custom_params = importlib.import_module(custom_params_file, package=None)
custom_params = custom_params.custom_params

for key in custom_params:
    logger.info("Setting {} to {} ".format(key, custom_params[key]))
    params[key] = custom_params[key]

# Shape precomputation dir.
parent_in_dir = params["masif_precomputation_dir"]
eval_list = []

all_roc_auc_scores = []

if len(sys.argv) == 3:
    # eval_list = [sys.argv[2].rstrip('_')]
    ppi_pair_ids = [sys.argv[2]]
# Read a list of pdb_chain entries to evaluate.
elif len(sys.argv) == 4 and sys.argv[2] == "-l":
    listfile = open(sys.argv[3])
    ppi_pair_ids = []
    for line in listfile:
        eval_list.append(line.rstrip())
    for mydir in os.listdir(parent_in_dir):
        ppi_pair_ids.append(mydir)
else:
    logger.warning("Not enough parameters")
    sys.exit(1)

logger.info(f"Evaluating predictions: {ppi_pair_ids}")

for ppi_pair_id in ppi_pair_ids:
    fields = ppi_pair_id.split("_")
    pdbid = fields[0]
    if len(fields) == 2 or fields[2] == "":
        chains = [ppi_pair_id.split("_")[1]]
    else:
        chains = [ppi_pair_id.split("_")[1], ppi_pair_id.split("_")[2]]

    if len(chains) == 1:
        pids = ["p1"]
    else:
        pids = ["p1", "p2"]

    for ix, pid in enumerate(pids):
        ply_file = masif_opts["ply_file_template"].format(pdbid, chains[ix])
        pdb_chain_id = pdbid + "_" + chains[ix]

        if (
            pdb_chain_id not in eval_list
            and pdb_chain_id + "_" not in eval_list
            and len(eval_list) > 0
        ):
            logger.warning(f"{pdb_chain_id} is not in eval list")
            continue

        try:
            p1 = pymesh.load_mesh(ply_file)
        except Exception as e:
            logger.exception("File does not exist: {}".format(ply_file), e)
            continue
        try:
            scores = np.load(
                params["out_pred_dir"] + "/pred_" + pdbid + "_" + chains[ix] + ".npy"
            )
        except Exception as e:
            logger.exception(
                "File does not exist: {}".format(
                    params["out_pred_dir"]
                    + "/pred_"
                    + pdbid
                    + "_"
                    + chains[ix]
                    + ".npy"
                ),
                e,
            )
            continue

        mymesh = p1

        ground_truth = mymesh.get_attribute("vertex_iface")
        # Compute ROC AUC for this protein.
        try:
            ground_truth = np.nan_to_num(ground_truth)
            scores = np.nan_to_num(scores)

            if len(np.unique(ground_truth)) > 1:
                roc_auc = roc_auc_score(ground_truth, scores[0])
                all_roc_auc_scores.append(roc_auc)
                logger.info(
                    "ROC AUC score for protein {} : {:.2f} ".format(
                        pdbid + "_" + chains[ix], roc_auc
                    )
                )
            else:
                logger.warning(
                    "Only one class present in ground truth for protein {}. ROC AUC score cannot be calculated.".format(
                        pdbid + "_" + chains[ix]
                    )
                )
                logger.warning("Ground truth: {}".format(ground_truth))
        except Exception as e:
            logger.exception("An error occurred while calculating ROC AUC score.", e)

        mymesh.remove_attribute("vertex_iface")
        mymesh.add_attribute("iface")
        mymesh.set_attribute("iface", scores[0])
        mymesh.remove_attribute("vertex_x")
        mymesh.remove_attribute("vertex_y")
        mymesh.remove_attribute("vertex_z")
        mymesh.remove_attribute("face_vertex_indices")

        if not os.path.exists(params["out_surf_dir"]):
            os.makedirs(params["out_surf_dir"])

        pymesh.save_mesh(
            params["out_surf_dir"] + pdb_chain_id + ".ply",
            mymesh,
            *mymesh.get_attribute_names(),
            use_float=True,
            ascii=True,
        )
        logger.info(
            "Successfully saved file " + params["out_surf_dir"] + pdb_chain_id + ".ply"
        )

med_roc = np.median(all_roc_auc_scores)

if len(all_roc_auc_scores) > 0:
    logger.info("Computed the ROC AUC for {} proteins".format(len(all_roc_auc_scores)))
    logger.info("Median ROC AUC score: {}".format(med_roc))
