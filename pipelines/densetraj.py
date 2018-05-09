import os
import multiprocessing

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

from cilia.misc import humansize
from visualize import animate


def check_all_same(arr):
    """
    If all items in arr are the same, return the item. Otherwise raise an exception.

    Parameters
    ----------
    arr : sequence
        Sequence where all elements are the same.

    Returns
    -------
    val : scalar
        The element that is repeated in arr.
    """
    uniq = np.unique(arr)
    if len(uniq) != 1:
        raise ValueError('all items in arr should be equal')
    return uniq[0]


class DBSCANCodebook:
    def __init__(self, descriptors):
        self.descriptors = descriptors
        self.clusters = None
        # self.codebooks = None

    def fit(self, tracks, progress=False):
        """
        Cluster the trajectories in each ROI.

        Parameters
        ----------
        tracks : structured ndarray
            Dense trajectories with fields 'roi' and descriptors.
        progress : bool, optional
            If True, show a progress bar.

        Returns
        -------
        self
        """
        # Cluster trajectories in each ROI independently
        self.clusters = dict()
        for desc in self.descriptors:
            self.clusters[desc] = self._cluster(tracks[desc], tracks['roi'])

        # Choose representative feature from each cluster (e.g. median)
        # self.codebooks = dict()
        # for desc in self.descriptors:
        #     codebook = {'cluster_centers': [], 'y_true': []}
        #
        #     if clusters[desc]['n_clusters'] > 0:
        #         for i in range(clusters[desc]['n_clusters']):
        #             rep = np.median(clusters[desc]['vectors'][i], axis=0)  # representative of features in cluster
        #             codebook['cluster_centers'].append(rep)
        #             codebook['y_true'].append(clusters[desc]['y_true'][i])
        #
        #     codebook = {k: np.array(v) for k, v in codebook.items()}
        #     self.codebooks[desc] = codebook

        return self

    def predict(self, tracks):
        """
        Predict the closest cluster each trajectory in tracks belongs to
        for each descriptor type.

        In the vector quantization literature, centroids is called the
        code book and each value returned by predict is the index of
        the closest code in the code book.

        Parameters
        ----------
        tracks : stuctured ndarray
            Trajectories to predict.

        Returns
        -------
        labels : dict of array
            Index of the cluster each trajectory belongs to for each descriptor type.
        """
        labels = dict()

        for desc in self.descriptors:
            if len(self.codebooks[desc]) > 0:
                cb_indices, dist = pairwise_distances_argmin_min(
                    tracks[desc], self.codebooks[desc]['cluster_centers'], metric='euclidean')
            else:
                cb_indices = np.full(len(tracks), fill_value=-1, dtype=np.int)

            labels[desc] = cb_indices

        return labels

    def _cluster(self, track_desc, rois):
        """
        Cluster the trajectory descriptors in each ROI independently.

        Parameters
        ----------
        track_desc : ndarray
            Trajectory descriptors to cluster.
        rois : ndarray
            The ROIs of the trajectories.

        Returns
        -------
        labels : ndarray of int
            Cluster labels for each trajectory. A label of -1 means that a trajectory does not belong
            to any cluster.
        """
        labels = np.full((len(track_desc),), dtype=np.int, fill_value=-1)

        for roi in np.unique(rois):
            indices = np.flatnonzero(rois == roi)  # indices of tracks in ROI
            D = pairwise.pairwise_distances(track_desc[indices], metric='euclidean')

            db = DBSCAN(eps=0.15, min_samples=5, metric='precomputed').fit(D)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True

            for l in np.unique(db.labels_):
                next_label = -1 if l == -1 else labels.max() + 1
                c_indices = indices[(db.labels_ == l) & core_samples_mask]  # indcices of samples in cluster
                labels[c_indices] = next_label

        return labels


def sample_tracks(targets, clusters, per_cluster, balance_targets=True):
    """

    Parameters
    ----------
    targets : sequence of int
        The 0/1 (normal/abnormal) labels of trajectories.
    clusters : sequence of int
        Cluster labels of trajectories. Trajectories not belonging
        to any clusters should have the label -1.
    per_cluster : int
        Maximum number of trajectories to sample from each cluster. If a cluster has
        less than the maximum, sample all of the trajectories in the cluster.
    balance_targets : bool, optional
        If True, the number of normal/abnormal targets will be balanced.

    Returns
    -------
    indices : ndarray of int
        The indices of the sampled trajectories.
    """
    assert len(targets) == len(clusters)
    n_tracks = len(targets)

    track_mask = np.zeros((n_tracks,), dtype=np.bool)

    # Choose a maximum of per_cluster trajectories in each cluster
    for cls in np.unique(clusters):
        if cls == -1:
            continue
        cls_indices = np.flatnonzero(clusters == cls)  # indices of tracks in cluster
        rand_indices = np.random.permutation(cls_indices)[:per_cluster]
        assert np.all(track_mask[rand_indices] == False)
        track_mask[rand_indices] = True

    #
    # Balance the data to have the same number of normal/abnormal trajectories. If there are more
    # normal trajectories than abnormal trajectories, add new abnormal trajectories to the data.
    # If there are more abnormal than normal trajectories, remove abnormal trajectories from data.
    #
    if balance_targets:
        target_counts = dict(zip(*np.unique(targets[track_mask], return_counts=True)))
        if target_counts[0] > target_counts[1]:  # more normal than abnormal
            #TODO: If more tracks should be sampled than exist
            tr_abnormal = np.flatnonzero(~track_mask & targets == 1)  # abnormal tracks not yet in sample
            rand = np.random.choice(tr_abnormal,
                                    size=min(len(tr_abnormal), target_counts[0]-target_counts[1]),
                                    replace=False)
            assert np.all(track_mask[rand] == False)
            track_mask[rand] = True
        else:
            quit('more abnormal than normal')

    return track_mask


def majority_vote(values):
    vals, counts = np.unique(values, return_counts=True)
    argmax = np.argmax(counts)
    # if np.count_nonzero(counts == counts[argmax]) > 1:
    #     return -1
    return vals[argmax]


def tracks_2_kernel(tracks_X, fields, tracks_Y=None, gamma=1.0, weights=None):
    """
    Compute kernel from trajectories.

    Parameters
    ----------
    tracks_X : structured array
    tracks_Y : strucutred array or None
    gamma : float
    fields : sequence of string
        Fields from trajectories to use for building kernel.
    weights : sequence of float or None
        Weights for fields.

    Returns
    -------

    """
    if not weights:
        weights = np.ones(shape=(len(fields),))
        weights = weights / np.sum(weights)
    if tracks_Y is None:
        tracks_Y = tracks_X

    K = np.zeros(shape=(len(tracks_X), len(tracks_Y)))
    print(humansize(K.nbytes))
    # print('Kernel {} - {}'.format(K.shape, humansize(K.nbytes)))

    for name, weight in zip(fields, weights):
        print('Kernel {} - {}'.format(K.shape, name))
        X = tracks_X[name]
        Y = tracks_Y[name]

        if name == 'trajectory':
            X = X.reshape((len(X), -1))
            Y = Y.reshape((len(Y), -1))
            old_min, old_max = -1, 1
            new_min, new_max = 0, 1
            X = ((X - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            Y = ((Y - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

        chi2 = pairwise.chi2_kernel(X, Y, gamma=gamma)
        # mu = 1.0 / kernel.mean()
        # K_train += weight * np.exp(-mu * kernel)
        K += weight * chi2
    return K


def cluster_tracks(tracks, progress_bar=False):
    """
    Compute clusters of trajectories for each ROI.
    """
    labels = np.full((len(tracks),), dtype=np.int, fill_value=-1)
    core_samples = np.zeros((len(tracks),), dtype=np.bool)

    #
    # Cluster trajectories for each ROI
    #
    uniq_rois = np.unique(tracks.roi)
    for i, roi in enumerate(uniq_rois):
        indices = np.flatnonzero(tracks.roi == roi)  # indices of tracks in ROI
        X = tracks[indices].trajectory.reshape((len(indices), -1))
        D = pairwise.pairwise_distances(X, metric='euclidean')

        db = DBSCAN(eps=0.15, min_samples=5, metric='precomputed').fit(D)

        for l in np.unique(db.labels_):
            if l == -1:
                new_l = -1
            else:
                new_l = labels.max() + 1
            labels[indices[db.labels_ == l]] = new_l
            core_samples[indices[db.core_sample_indices_]] = True

        if progress_bar:
            progress(i+1, len(uniq_rois))

    return labels


def _call_animate_tracks_grid(args):
    outfile, rois, roi_tracks, n_rows, n_cols = args
    animate.tracks_grid(outfile, rois, roi_tracks, n_rows, n_cols)
    return outfile


def visualize_patient_rois(tracks, output_dir, n_rows, n_cols):
    """
    For each patient, create video(s) showing the ROIs and their corresponding trajectories
    for that patient.

    Multiple videos will be created if all ROIs do not fit into the grid of a single video.

    Parameters
    ----------
    tracks : strucutred array
        Desnse trajectories
    output_dir : str
        Path to directory where output will be saved. Is created if it does not exist.

    Returns
    -------
    None
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fargs = []

    uniq_patients = np.unique(tracks['patient'])
    for i, pat in enumerate(uniq_patients):
        outfile = os.path.join(output_dir, '{}_roi_tracks.mp4'.format(pat))

        # rois belonging to patient
        rois = np.unique(tracks['video'][tracks['patient'] == pat])[:n_cols*n_rows]

        # tracks in each individual roi. Only extract attributes we need to plot trajectories
        roi_tracks = [tracks[['coords', 'frame_num']][tracks['video'] == roi] for roi in rois]

        # animate.tracks_grid(outfile, rois, roi_tracks, n_rows, n_cols)
        fargs.append((outfile, rois, roi_tracks, n_rows, n_cols))

    with multiprocessing.Pool() as pool:
        for i, retval in enumerate(pool.imap_unordered(_call_animate_tracks_grid, fargs)):
            print('{}/{}: {}'.format(i+1, len(fargs), retval))
