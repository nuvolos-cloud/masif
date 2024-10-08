import time
import os
import logging
from sklearn import metrics
import numpy as np
import tensorflow as tf
from masif_modules.MaSIF_site import MaSIF_site


logger = logging.getLogger(__name__)


# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


def pad_indices(indices, max_verts):
    padded_ix = np.zeros((len(indices), max_verts), dtype=int)
    for patch_ix in range(len(indices)):
        padded_ix[patch_ix] = np.concatenate(
            [indices[patch_ix], [patch_ix] * (max_verts - len(indices[patch_ix]))]
        )
    return padded_ix


# Run masif site on a protein, on a previously trained network.
def run_masif_site(
    params,
    learning_obj: MaSIF_site,
    rho_wrt_center,
    theta_wrt_center,
    input_feat,
    mask,
    indices,
):

    indices = pad_indices(indices, mask.shape[1])
    mask = np.expand_dims(mask, 2)
    feed_dict = {
        learning_obj.rho_coords: rho_wrt_center,
        learning_obj.theta_coords: theta_wrt_center,
        learning_obj.input_feat: input_feat,
        learning_obj.mask: mask,
        learning_obj.indices_tensor: indices,
    }

    score = learning_obj.session.run([learning_obj.full_score], feed_dict=feed_dict)
    return score


def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    if np.isnan(labels).any():
        logger.warning("Warning: labels contains NaN")
    if np.isnan(labels).any():
        logger.warning("Warning: score contains NaN")
    labels = np.nan_to_num(labels)
    dist_pairs = np.nan_to_num(dist_pairs)
    return metrics.roc_auc_score(labels, dist_pairs)


def train_masif_site(
    learning_obj: MaSIF_site,
    params,
    batch_size=100,
    num_iterations=100,
):
    writer = tf.summary.create_file_writer(params["tensorboard_log_dir"])

    # Open training list.

    list_training_loss = []
    list_training_auc = []
    best_val_auc = 0

    out_dir = params["model_dir"]
    logger.info("Training parameters:\n")
    for key in params:
        logger.info("{}: {}\n".format(key, params[key]))

    training_list = open(params["training_list"]).readlines()
    training_list = [x.rstrip() for x in training_list]

    testing_list = open(params["testing_list"]).readlines()
    testing_list = [x.rstrip() for x in testing_list]

    data_dirs = os.listdir(params["masif_precomputation_dir"])

    if "exclude_list" in params:
        exclude_list = open(params["exclude_list"]).readlines()
    else:
        exclude_list = []
    exclude_list = [x.rstrip() for x in exclude_list]
    exclude_set = set(exclude_list)

    for num_iter in range(num_iterations):
        np.random.shuffle(data_dirs)
        n_val = len(data_dirs) // 10
        val_dirs = set(data_dirs[(len(data_dirs) - n_val) :])
        # Start training epoch:
        list_training_loss = []
        list_training_auc = []
        list_val_auc = []
        list_val_pos_labels = []
        list_val_neg_labels = []
        list_val_names = []

        logger.info("Starting epoch {}".format(num_iter))
        tic = time.time()
        all_training_labels = []
        all_training_scores = []
        all_val_labels = []
        all_val_scores = []
        all_test_labels = []
        all_test_scores = []
        count_proteins = 0

        list_test_auc = []
        list_test_names = []
        all_test_labels = []
        all_test_scores = []

        p = 0
        for ppi_pair_id in data_dirs:
            if ppi_pair_id in exclude_set:
                logger.info(f"Skipping {ppi_pair_id} because it is in exclude_set")
                continue
            p += 1
            mydir = params["masif_precomputation_dir"] + ppi_pair_id + "/"
            pdbid = ppi_pair_id.split("_")[0]
            chains1 = ppi_pair_id.split("_")[1]
            logger.info(f"Processing protein no. {p}:  {ppi_pair_id}")
            if len(ppi_pair_id.split("_")) > 2:
                chains2 = ppi_pair_id.split("_")[2]
            else:
                chains2 = ""
            pids = []
            if pdbid + "_" + chains1 in training_list:
                pids.append("p1")
            if pdbid + "_" + chains2 in training_list:
                pids.append("p2")
            for pid in pids:
                try:
                    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                except Exception as e:
                    logger.info(f"Error loading {mydir + pid + '_iface_labels.npy'}", e)
                    continue
                if len(iface_labels) > 8000:
                    logger.info(
                        f"Skipping {mydir + pid + '_iface_labels.npy'} because it has too many labels"
                    )
                    exclude_set.add(ppi_pair_id)
                    continue
                if (
                    np.sum(iface_labels) > 0.75 * len(iface_labels)
                    or np.sum(iface_labels) < 30
                ):
                    logger.info(
                        f"Skipping {mydir + pid + '_iface_labels.npy'} because label sum is out of bounds"
                    )
                    exclude_set.add(ppi_pair_id)
                    continue
                count_proteins += 1

                rho_wrt_center = np.load(mydir + pid + "_rho_wrt_center.npy")
                theta_wrt_center = np.load(mydir + pid + "_theta_wrt_center.npy")
                input_feat = np.load(mydir + pid + "_input_feat.npy")
                if np.sum(params["feat_mask"]) < 5:
                    input_feat = mask_input_feat(input_feat, params["feat_mask"])
                mask = np.load(mydir + pid + "_mask.npy")
                mask = np.expand_dims(mask, 2)
                indices = np.load(mydir + pid + "_list_indices.npy", encoding="latin1")
                # indices is (n_verts x <30), it should be
                indices = pad_indices(indices, mask.shape[1])
                tmp = np.zeros((len(iface_labels), 2))
                for i in range(len(iface_labels)):
                    if iface_labels[i] == 1:
                        tmp[i, 0] = 1
                    else:
                        tmp[i, 1] = 1
                iface_labels_dc = tmp

                pos_labels = np.where(iface_labels == 1)[0]
                neg_labels = np.where(iface_labels == 0)[0]
                np.random.shuffle(neg_labels)
                np.random.shuffle(pos_labels)
                # Scramble neg idx, and only get as many as pos_labels to balance the training.
                if params["n_conv_layers"] == 1:
                    n = min(len(pos_labels), len(neg_labels))
                    n = min(n, batch_size // 2)
                    subset = np.concatenate([neg_labels[:n], pos_labels[:n]])

                    rho_wrt_center = rho_wrt_center[subset]
                    theta_wrt_center = theta_wrt_center[subset]
                    input_feat = input_feat[subset]
                    mask = mask[subset]
                    iface_labels_dc = iface_labels_dc[subset]
                    indices = indices[subset]
                    pos_labels = range(0, n)
                    neg_labels = range(n, n * 2)
                else:
                    n = min(len(pos_labels), len(neg_labels))
                    neg_labels = neg_labels[:n]
                    pos_labels = pos_labels[:n]

                feed_dict = {
                    learning_obj.rho_coords: rho_wrt_center,
                    learning_obj.theta_coords: theta_wrt_center,
                    learning_obj.input_feat: input_feat,
                    learning_obj.mask: mask,
                    learning_obj.labels: iface_labels_dc,
                    learning_obj.pos_idx: pos_labels,
                    learning_obj.neg_idx: neg_labels,
                    learning_obj.indices_tensor: indices,
                }

                if ppi_pair_id in val_dirs:
                    logger.info("Validating on {} {}\n".format(ppi_pair_id, pid))
                    feed_dict[learning_obj.keep_prob] = 1.0
                    training_loss, score, eval_labels = learning_obj.session.run(
                        [
                            learning_obj.data_loss,
                            learning_obj.eval_score,
                            learning_obj.eval_labels,
                        ],
                        feed_dict=feed_dict,
                    )
                    # Log validation loss to TensorBoard
                    with writer.as_default():
                        tf.summary.scalar(
                            "validation_loss", np.mean(training_loss), step=num_iter
                        )
                    if np.isnan(eval_labels).any():
                        exclude_set.add(ppi_pair_id)
                        logger.warning(
                            f"Reloading last training checkpoint as eval label for {ppi_pair_id} had NaN..."
                        )
                        learning_obj.saver.restore(
                            learning_obj.session, out_dir + "model"
                        )
                        continue
                    elif np.isnan(score).any():
                        exclude_set.add(ppi_pair_id)
                        logger.warning(
                            f"Reloading last training checkpoint as eval score for {ppi_pair_id} had NaN..."
                        )
                        learning_obj.saver.restore(
                            learning_obj.session, out_dir + "model"
                        )
                        continue
                    else:
                        learning_obj.saver.save(learning_obj.session, out_dir + "model")
                    eval_labels = np.nan_to_num(eval_labels)
                    score = np.nan_to_num(score)
                    auc = metrics.roc_auc_score(eval_labels[:, 0], score)
                    list_val_pos_labels.append(np.sum(iface_labels))
                    list_val_neg_labels.append(len(iface_labels) - np.sum(iface_labels))
                    list_val_auc.append(auc)
                    list_val_names.append(ppi_pair_id)
                    all_val_labels = np.concatenate([all_val_labels, eval_labels[:, 0]])
                    all_val_scores = np.concatenate([all_val_scores, score])
                else:
                    logger.info("Training on {} {}\n".format(ppi_pair_id, pid))
                    feed_dict[learning_obj.keep_prob] = 1.0
                    (
                        _,
                        training_loss,
                        norm_grad,
                        score,
                        eval_labels,
                    ) = learning_obj.session.run(
                        [
                            learning_obj.optimizer,
                            learning_obj.data_loss,
                            learning_obj.norm_grad,
                            learning_obj.eval_score,
                            learning_obj.eval_labels,
                        ],
                        feed_dict=feed_dict,
                    )
                    # Log training loss and gradient norm to TensorBoard
                    with writer.as_default():
                        tf.summary.scalar(
                            "training_loss", np.mean(training_loss), step=num_iter
                        )
                        tf.summary.scalar("norm_grad", norm_grad, step=num_iter)

                    if np.isnan(eval_labels).any():
                        exclude_set.add(ppi_pair_id)
                        logger.warning(
                            f"Reloading last training checkpoint as training label for {ppi_pair_id} had NaN..."
                        )
                        learning_obj.saver.restore(
                            learning_obj.session, out_dir + "model"
                        )
                        continue
                    elif np.isnan(score).any():
                        exclude_set.add(ppi_pair_id)
                        logger.warning(
                            f"Reloading last training checkpoint as training score for {ppi_pair_id} had NaN..."
                        )
                        learning_obj.saver.restore(
                            learning_obj.session, out_dir + "model"
                        )
                        continue
                    else:
                        learning_obj.saver.save(learning_obj.session, out_dir + "model")
                    eval_labels = np.nan_to_num(eval_labels)
                    score = np.nan_to_num(score)
                    all_training_labels = np.concatenate(
                        [all_training_labels, eval_labels[:, 0]]
                    )
                    all_training_scores = np.concatenate([all_training_scores, score])
                    auc = metrics.roc_auc_score(eval_labels[:, 0], score)
                    list_training_auc.append(auc)
                    list_training_loss.append(np.mean(training_loss))

        # Run testing cycle.
        for ppi_pair_id in data_dirs:
            mydir = params["masif_precomputation_dir"] + ppi_pair_id + "/"
            logger.info(f"Running test cycle for {ppi_pair_id}")
            pdbid = ppi_pair_id.split("_")[0]
            chains1 = ppi_pair_id.split("_")[1]
            if len(ppi_pair_id.split("_")) > 2:
                chains2 = ppi_pair_id.split("_")[2]
            else:
                chains2 = ""
            pids = []
            if pdbid + "_" + chains1 in testing_list:
                pids.append("p1")
            if pdbid + "_" + chains2 in testing_list:
                pids.append("p2")
            for pid in pids:
                logger.info("Testing on {} {}\n".format(ppi_pair_id, pid))
                try:
                    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                except Exception as e:
                    logger.info(f"Error loading {mydir + pid + '_iface_labels.npy'}", e)
                    continue
                if len(iface_labels) > 20000:
                    logger.info(
                        f"Skipping {mydir + pid + '_iface_labels.npy'} because it has too many labels"
                    )
                    exclude_set.add(ppi_pair_id)
                    continue
                if (
                    np.sum(iface_labels) > 0.75 * len(iface_labels)
                    or np.sum(iface_labels) < 30
                ):
                    logger.info(
                        f"Skipping {mydir + pid + '_iface_labels.npy'} because label sum is out of bounds"
                    )
                    exclude_set.add(ppi_pair_id)
                    continue
                count_proteins += 1

                rho_wrt_center = np.load(mydir + pid + "_rho_wrt_center.npy")
                theta_wrt_center = np.load(mydir + pid + "_theta_wrt_center.npy")
                input_feat = np.load(mydir + pid + "_input_feat.npy")
                if np.sum(params["feat_mask"]) < 5:
                    input_feat = mask_input_feat(input_feat, params["feat_mask"])
                mask = np.load(mydir + pid + "_mask.npy")
                mask = np.expand_dims(mask, 2)
                indices = np.load(mydir + pid + "_list_indices.npy", encoding="latin1")
                # indices is (n_verts x <30), it should be
                indices = pad_indices(indices, mask.shape[1])
                tmp = np.zeros((len(iface_labels), 2))
                for i in range(len(iface_labels)):
                    if iface_labels[i] == 1:
                        tmp[i, 0] = 1
                    else:
                        tmp[i, 1] = 1
                iface_labels_dc = tmp

                pos_labels = np.where(iface_labels == 1)[0]
                neg_labels = np.where(iface_labels == 0)[0]

                feed_dict = {
                    learning_obj.rho_coords: rho_wrt_center,
                    learning_obj.theta_coords: theta_wrt_center,
                    learning_obj.input_feat: input_feat,
                    learning_obj.mask: mask,
                    learning_obj.labels: iface_labels_dc,
                    learning_obj.pos_idx: pos_labels,
                    learning_obj.neg_idx: neg_labels,
                    learning_obj.indices_tensor: indices,
                }

                feed_dict[learning_obj.keep_prob] = 1.0
                score = learning_obj.session.run(
                    [learning_obj.full_score], feed_dict=feed_dict
                )
                score = score[0]
                if np.isnan(iface_labels).any():
                    logger.warning(
                        f"Warning: iface_labels contains NaN for {ppi_pair_id} {pid}"
                    )
                    exclude_set.add(ppi_pair_id)
                if np.isnan(score).any():
                    logger.warning(
                        f"Warning: score contains NaN for {ppi_pair_id} {pid}"
                    )
                    exclude_set.add(ppi_pair_id)
                    logger.warning(
                        "Reloading last training checkpoint as testing score had NaN..."
                    )
                    learning_obj.saver.restore(learning_obj.session, out_dir + "model")
                iface_labels = np.nan_to_num(iface_labels)
                score = np.nan_to_num(score)
                auc = metrics.roc_auc_score(iface_labels, score)
                list_test_auc.append(auc)
                list_test_names.append((ppi_pair_id, pid))
                all_test_labels.append(iface_labels)
                all_test_scores.append(score)

        outstr = "Epoch ran on {} proteins\n".format(count_proteins)
        outstr += "Per protein AUC mean (training): {:.4f}; median: {:.4f} for iter {}\n".format(
            np.mean(list_training_auc), np.median(list_training_auc), num_iter
        )

        outstr += "Per protein AUC mean (validation): {:.4f}; median: {:.4f} for iter {}\n".format(
            np.mean(list_val_auc), np.median(list_val_auc), num_iter
        )
        outstr += (
            "Per protein AUC mean (test): {:.4f}; median: {:.4f} for iter {}\n".format(
                np.mean(list_test_auc), np.median(list_test_auc), num_iter
            )
        )

        if len(all_test_labels) > 0 and len(all_test_scores) > 0:
            flat_all_test_labels = np.concatenate(all_test_labels, axis=0)
            flat_all_test_scores = np.concatenate(all_test_scores, axis=0)
            flat_all_test_labels = np.nan_to_num(flat_all_test_labels)
            flat_all_test_scores = np.nan_to_num(flat_all_test_scores)
            outstr += "Testing auc (all points): {:.2f}".format(
                metrics.roc_auc_score(flat_all_test_labels, flat_all_test_scores)
            )
        else:
            outstr += "Testing auc (all points): N/A"

        outstr += "Epoch took {:2f}s\n".format(time.time() - tic)
        logger.info(outstr)

        if np.mean(list_val_auc) > best_val_auc:
            logger.info(
                ">>> Saving model. AUC improved from {} to {}\n".format(
                    best_val_auc, np.mean(list_val_auc)
                )
            )
            best_val_auc = np.mean(list_val_auc)
            output_model = out_dir + "model"
            learning_obj.saver.save(learning_obj.session, output_model)
            # Save the scores for test.
            flat_all_test_labels = np.concatenate(all_test_labels, axis=0)
            flat_all_test_scores = np.concatenate(all_test_scores, axis=0)
            flat_all_test_labels = np.nan_to_num(flat_all_test_labels)
            flat_all_test_scores = np.nan_to_num(flat_all_test_scores)
            np.save(out_dir + "test_labels.npy", flat_all_test_labels)
            np.save(out_dir + "test_scores.npy", flat_all_test_scores)
            np.save(out_dir + "test_names.npy", list_test_names)
        elif round(np.mean(list_training_auc), 3) == 0.500:
            logger.warning(
                "Reloading last training checkpoint as iteration AUC score was 0.500..."
            )
            learning_obj.saver.restore(learning_obj.session, out_dir + "model")

    # logger.info("Training finished, saving model to: {}".format(out_dir + "model"))
    # tf.saved_model.save(learning_obj.session, out_dir + "model")
    # logger.info("Model saved.")
