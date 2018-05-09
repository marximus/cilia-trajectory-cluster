import copy

import numpy as np
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, MeanShift
from scipy.spatial.distance import squareform, pdist
import hdbscan


def cluster(features, method, **kwargs):
    """Cluster tracklet features.

    Parameters
    ----------
    features : structured ndarray
        Tracklet features as a structured numpy array where the field names
    method
    kwargs

    Returns
    -------
    clusters : ndarray of int
        Cluster label of each tracklet.

    References
    ----------
    Anjum, Nadeem, and Andrea Cavallaro. "Multifeature object trajectory clustering for video analysis." IEEE
    Transactions on Circuits and Systems for Video Technology 18.11 (2008): 1555-1564.
    """
    n_tracklets = len(features)

    feature_labels = np.empty(n_tracklets, dtype=[(name, np.int) for name in features.dtype.names])

    for feature_name in features.dtype.names:
        X = features[feature_name]
        # normalize feature space
        X = preprocessing.scale(X)

        # cluster using distance matrix
        if method == 'dbscan':
            dist = squareform(pdist(X, 'euclidean'))
            db = DBSCAN(metric='precomputed', **kwargs).fit(dist)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            # only use core samples for clusters
            # labels[~core_samples_mask] = -1
        elif method == 'hdbscan':
            dist = squareform(pdist(X, 'euclidean'))
            clusterer = hdbscan.HDBSCAN(metric='precomputed', **kwargs)
            clusterer.fit(dist)
            labels = clusterer.labels_
        elif method == 'meanshift':
            # TODO: Error if two or three tracklets in ROI.
            ms = MeanShift(**kwargs).fit(X)
            labels = ms.labels_
        else:
            raise ValueError('cluster_algorithm must be in {}'.format(
                ('hdbscan', 'dbscan', 'meanshift')))

        feature_labels[feature_name] = labels

    return feature_labels


def fuse_clusters(feature_clusters, ):
    """
    Fuse clustering results across all feature spaces and return cluster labels.

    Parameters
    ----------
    feature_clusters : structured ndarray
        Structured numpy array where each field defines a feature. The field
        names (ndarray.dtype.names) should contain the name of the features.

    Returns
    -------
    clusters : ndarray of int
        Fused cluster labels for each point.

    References
    ----------
    """
    # TODO: Check that features.dtype.names ordering does not change
    n_clusters = np.array([len(set(feature_clusters[k])) - (1 if -1 in feature_clusters[k] else 0)
                           for k in feature_clusters.dtype.names])
    median_idx = np.argsort(n_clusters)[len(n_clusters)//2]
    init_feat_name = feature_clusters.dtype.names[median_idx]
    print('init_feat_name = {}'.format(init_feat_name))
    print(dict(zip(feature_clusters.dtype.names, n_clusters)), end='\n\n')

    # map cluster ids to indices of tracklets in cluster (for every feature space)
    feature_labels_map = {name: dict(zip(*unique_with_indices(feature_clusters[name])))
                          for name in feature_clusters.dtype.names}

    clusters = copy.deepcopy(feature_labels_map[init_feat_name])

    for feat_name in set(feature_clusters.dtype.names) - set((init_feat_name,)):
        for cid1 in clusters:
            max_intersect = set()

            for cid2 in feature_labels_map[feat_name]:
                intersect = clusters[cid1] & feature_labels_map[feat_name][cid2]
                if len(intersect) > len(max_intersect):
                    max_intersect = intersect

            clusters[cid1] = max_intersect

    # convert to array of ints where values are the cluster ids for the tracklets
    clusters_final = np.full(len(feature_clusters), -1, dtype=np.int)
    for cluster_id, track_ids in clusters.items():
        clusters_final[list(track_ids)] = cluster_id

    return clusters_final


def unique_with_indices(data):
    """
    Return unique values along with the indices where each unique value occurs.
    """
    unq, unq_inv, unq_cnt = np.unique(data, return_counts=True, return_inverse=True)
    indices = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
    return unq, list(map(set, indices))


