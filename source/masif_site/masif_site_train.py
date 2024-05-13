# Header variables and parameters.
import os
import importlib
import sys
import logging
import numpy as np
from default_config.masif_opts import masif_opts
import tensorflow as tf

"""
masif_site_train.py: Entry function to train MaSIF-site.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

logger = logging.getLogger(__name__)

# Use keras 2.16 instead of keras 3 for tf.compat.v1.layers.dense()
os.environ["TF_USE_LEGACY_KERAS"] = "1"
tf.compat.v1.disable_v2_behavior()
params = masif_opts["site"]

if len(sys.argv) > 1:
    custom_params_file = sys.argv[1]
    custom_params = importlib.import_module(custom_params_file, package=None)
    custom_params = custom_params.custom_params

    for key in custom_params:
        logger.info("Setting {} to {} ".format(key, custom_params[key]))
        params[key] = custom_params[key]


# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


if "pids" not in params:
    params["pids"] = ["p1", "p2"]

# Build the neural network model
from masif_modules.MaSIF_site import MaSIF_site

if "n_theta" in params:
    learning_obj = MaSIF_site(
        params["max_distance"],
        n_thetas=params["n_theta"],
        n_rhos=params["n_rho"],
        n_rotations=params["n_rotations"],
        idx_gpu="/gpu:1",
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )
else:
    learning_obj = MaSIF_site(
        params["max_distance"],
        n_thetas=4,
        n_rhos=3,
        n_rotations=4,
        idx_gpu="/gpu:1",
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )

# train
from masif_modules.train_masif_site import train_masif_site

logger.debug(params["feat_mask"])
if not os.path.exists(params["model_dir"]):
    os.makedirs(params["model_dir"])
else:
    # Load existing network.
    logger.info(f"Reading pre-trained network from {params['model_dir']+'model'}")
    learning_obj.saver.restore(learning_obj.session, params["model_dir"] + "model")

train_masif_site(learning_obj, params)
