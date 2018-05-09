import os
import shutil
import sys
import pprint
import tempfile
import pprint
import copy
from io import FileIO
from os.path import basename, splitext

import numpy as np
import pandas as pd
import hdbscan
from PyPDF2 import PdfFileMerger
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, MeanShift
from scipy.spatial.distance import squareform, pdist
from joblib import Parallel, delayed
from scipy import misc
import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from cilia import datasets, tree
from tracklet.tracklet import features as trackfeatures
from trackletviz.trackletviz import static as trackplot
from trackletviz.trackletviz import animate as trackanimate

# FOR CHIMAY
sys.path.append("/home/mam588/dense_trajectory_release_v1.2")
# FOR VIVALDI
sys.path.append("/home/mam588/dense_trajectory_release_v1.2")
DATA_PREFIX = "/ssd/Cohorts"


import densetrack


TRAIN_DATA = (os.path.join(DATA_PREFIX, "CHP"),
              os.path.join(DATA_PREFIX, "CNMC"))
TEST_DATA = os.path.join(DATA_PREFIX, "Ashok_Eileen")

cachedir = "/ssd/cilia_data_cache"
track_kwargs = {'track_length': 10, 'min_distance': 5, 'patch_size': 32,
                'poly_n': 5, 'poly_sigma': 0.5, 'scale_num': 1}

BASE_OUTPUT_DIR = os.path.expanduser('~/track_output')

# create cache directory if it doesn't exist
tracks_cachedir = os.path.join(cachedir, 'dense_tracks')
if not os.path.exists(tracks_cachedir):
    os.mkdir(tracks_cachedir)

# create directory for output
OVERWRITE = True
if os.path.exists(BASE_OUTPUT_DIR) and OVERWRITE:
    shutil.rmtree(BASE_OUTPUT_DIR)
if not os.path.exists(BASE_OUTPUT_DIR):
    os.mkdir(BASE_OUTPUT_DIR)


def dict_to_string(d):
    """Return a string representation of the key/value pairs in dictionary."""
    s = '_'.join(['{}-{}'.format(k, d[k]) for k in sorted(d)])
    s = s.replace(' ', '')
    return s


plt.style.use(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mplstyle', 'paper.mplstyle'))


if __name__ == "__main__":
    ###############################################################
    # Load rois and compute the trajectories for each roi
    ###############################################################
    rois = datasets.load_rois(TRAIN_DATA[0], TRAIN_DATA[1])[:50]
    tracks = densetrack.compute(rois, tracks_cachedir, show_progress=True,
                                inherit_fields=['patient', 'video', 'target'], **track_kwargs)

    roi_unq, unq_inv, unq_cnt = np.unique(tracks['video'], return_counts=True, return_inverse=True)
    roi_unq_tracks = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))


    ####################################################################################
    # Save 2d tracklet animations.
    ####################################################################################
    # create output directory in which to save animations if DNE
    if not os.path.exists(os.path.join(BASE_OUTPUT_DIR, 'track_anim2d')):
        os.mkdir(os.path.join(BASE_OUTPUT_DIR, 'track_anim2d'))

    # for video, inds in zip(roi_unq, roi_unq_tracks):
    #     print(video)
    #     # print(np.load(video).shape)
    #     # print(len(inds))
    #     # print()
    #     trackanimate.trajectory_2d(
    #         tracks['coords'][inds],
    #         tracks['frame_num'][inds],
    #         video,
    #         save_path=os.path.join(BASE_OUTPUT_DIR, 'track_anim2d', splitext(basename(video))[0]+'.mp4'),
    #         bitrate=5000, fps=5,
    #         line_kws=dict(linewidths=1.0, colors='g'),
    #         # scat_kws=dict(s=5, c='r', marker='+')
    #     )
    # quit()

    Parallel(n_jobs=-1, verbose=100)(delayed(trackanimate.trajectory_2d)(
        tracks['coords'][inds], tracks['frame_num'][inds], video,
        save_path=os.path.join(BASE_OUTPUT_DIR, 'track_anim2d', splitext(basename(video))[0]+'.mp4'),
        fps=10, bitrate=5000,
        line_kws=dict(linewidths=1.0, colors='g'),
        scat_kws=dict(s=5, c='r', marker='+')
    ) for video, inds in zip(roi_unq, roi_unq_tracks))
    quit()

    # ####################################################################################
    # # Save 3d tracklet animations.
    # ####################################################################################
    # # create output directory in which to save animations if DNE
    # if not os.path.exists(os.path.join(BASE_OUTPUT_DIR, 'track_anim')):
    #     os.mkdir(os.path.join(BASE_OUTPUT_DIR, 'track_anim'))
    #
    # Parallel(n_jobs=-1, verbose=100)(delayed(trackanimate.trajectory_3d)(
    #     tracks['coords'][inds], tracks['frame_num'][inds], video,
    #     os.path.join(BASE_OUTPUT_DIR, 'track_anim', splitext(basename(video))[0]+'.mp4')
    # ) for video, inds in zip(roi_unq, roi_unq_tracks))

    ####################################################################################
    # Plot velocity magnitudes for each ROI
    ####################################################################################
    velocity = trackfeatures.velocity(tracks['coords'])
    velocity_mag = np.sqrt(velocity[:, :, 0]**2, + velocity[:, :, 1]**2)

    n_lines, n_pts, _ = tracks['coords'].shape
    frame_num = np.mgrid[:n_lines, :n_pts][1] - n_pts + 1 + tracks['frame_num'][:, None]

    data = {'roi': [], 'vel_mag': [], 'frame_num': [], 'target': []}
    for roi in np.unique(tracks['video']):
        ind = np.flatnonzero(tracks['video'] == roi)
        target = rois[rois['video'] == roi]['target'].iloc[0]
        target = {0:'normal', 1:'abnormal'}[target]

        data['roi'].extend([os.path.splitext(os.path.basename(roi))[0]]*len(ind)*n_pts)
        data['target'].extend([target]*len(ind)*n_pts)
        data['vel_mag'].extend(list(velocity_mag[ind].flatten()))
        data['frame_num'].extend(list(frame_num[ind].flatten()))
    df = pd.DataFrame(data)
    df = df[df['frame_num'] <= 100]

    sns.set(style='ticks')
    with sns.plotting_context('paper', font_scale=0.7):
        g = sns.factorplot(
            x='frame_num', y='vel_mag', hue='target', data=df, kind='point',
            palette=dict(normal='green', abnormal='red'),
            col='roi', col_wrap=2, size=2, aspect=3,
            scale=0.3, errwidth=0.8, # will be passed to underlying plot function
        )
        g.savefig(os.path.join(BASE_OUTPUT_DIR, 'velocity_mag.pdf'))

    quit()


    ##################################################
    # Compute tracklet features
    ##################################################
    # print('Computing tracklet features ...')
    # track_features = dict(
    #     velocity=trackfeatures.velocity(tracks['coords']),
    #     # curvature=trackfeatures.curvature(tracks['coords'][:, :, 0], tracks['coords'][:, :, 1]),
    #     average_velocity=trackfeatures.average_velocity(tracks['coords']),
    #     directional_distance=trackfeatures.directional_distance(tracks['coords']),
    #     trajectory_mean=trackfeatures.trajectory_mean(tracks['coords']),
    #     # directional_histogram=trackfeatures.directional_histogram(tracks['coords'], 10)[0],
    # )
    # names = list(track_features.keys())
    # types = [track_features[name].dtype for name in names]
    # shapes = [track_features[name].shape[1:] for name in names]
    # new_track_features = np.empty(len(tracks), dtype=list(zip(names, types, shapes)))
    # for name in track_features:
    #     new_track_features[name] = track_features[name]
    # track_features = new_track_features


    ############################################################################################
    # Cluster tracklets in each feature space separately.
    # The tracklets in each ROI are clustered separately.
    ############################################################################################
    from tracklet.tracklet import distance as trackdist
    from tracklet.tracklet import cluster as trackcluster

    print('Clustering tracklet features ...')

    labels_feature = np.empty(len(tracks), dtype=[(name, np.int) for name in track_features.dtype.names])
    track_labels = np.empty(len(tracks), dtype=np.int)

    curr_label = 0
    for i, (roi, track_inds) in enumerate(zip(roi_unq, roi_unq_tracks)):
        print('Clustering {}/{} - {} tracklets'.format(i+1, len(roi_unq), len(track_inds)))

        # handle case where ROI has small amount of tracklets
        if len(track_inds) <= 3:
            labels_feature[track_inds] = tuple(-1 for _ in labels_feature.dtype.names)
            track_labels[track_inds] = -1
            continue

        # cluster tracklets in each feature space seperately
        labels_feat = trackcluster.cluster(
            track_features[track_inds], 'meanshift', bandwidth=None, cluster_all=False)
        # fuse clusters across all feature spaces
        labels = trackcluster.fuse_clusters(labels_feat)

        # X = tracks['trajectory'][track_inds]
        # dist = trackdist.euclidean_sum(X, X)
        # db = DBSCAN(metric='precomputed', eps=0.05, min_samples=15).fit(dist)
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # labels = db.labels_

        # update global labels
        labels_feature[track_inds] = labels_feat
        for label in np.unique(labels):
            if label == -1:
                new_label = -1
            else:
                new_label = curr_label
                curr_label += 1
            track_labels[track_inds[labels == label]] = new_label


    #####################################################################################
    # Visualize clustered tracklets in each feature space separately. A pdf page is saved
    # for each ROI in a temporary directory and then merged into a single pdf.
    #####################################################################################
    outfile = os.path.join(BASE_OUTPUT_DIR, "tracklet_features.pdf")
    print('Saving {} ...'.format(outfile))

    feature_plot = {
        'average_velocity': dict(title='Average Velocity', xlabel='Horizontal component', ylabel='Vertical Component'),
        'directional_distance': dict(title='Directional distance', xlabel='Horizontal length', ylabel='Vertical length'),
        'trajectory_mean': dict(title='Trajectory mean', xlabel='Horizontal component', ylabel='Vertical Component'),
        # 'directional_histogram': dict(title='Directional histogram')
    }

    # seaborn.jointplot does not allow an axes instance to be supplied. It creates its own
    # figure and axes internally and returns the newly created axes. In order to use
    # seaborn.jointplot in an existing axes, first save the plot as an image and then load
    # it into the axes.
    with tempfile.TemporaryDirectory() as tmpdir:
        for roi_idx, (roi, track_inds) in enumerate(zip(tqdm.tqdm(roi_unq), roi_unq_tracks)):
            fname = os.path.splitext(os.path.basename(roi))[0]

            fig = plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(len(feature_plot), 4)
                                   # left=0.01, bottom=0.01, right=0.99, top=0.99, hspace=0.1, wspace=0.01)

            for feat_idx, (feat_name, params) in enumerate(feature_plot.items()):
                # there is a problem with numpy arrays containing single elements so for now just duplicate the
                # element to create an array of length 2 if a length 1 array is present
                if len(track_inds) == 1:
                    track_inds = np.repeat(track_inds, 2)

                # color for each tracklet based on its cluster id
                cluster_ids = labels_feature[feat_name][track_inds]
                unq_clusters, inv = np.unique(cluster_ids, return_inverse=True)
                colors = np.array(sns.husl_palette(len(unq_clusters)))

                # plot tracklets
                group_kws = {cluster_id: {'color': color} for cluster_id, color in zip(unq_clusters, colors)}
                ax1 = plt.subplot(gs[feat_idx, 1:], projection='3d')
                trackplot.tracklet_3d(tracks['coords'][track_inds], tracks['frame_num'][track_inds],
                                   ax=ax1, ms=0.3, lw=0.3, groups=cluster_ids, group_kwargs=group_kws)

                # save plot as image to file
                # see http://stackoverflow.com/questions/37945495/python-matplotlib-save-as-tiff
                # on how to save to memory rather than a file
                g = (sns.jointplot(track_features[feat_name][track_inds, 0],
                                   track_features[feat_name][track_inds, 1],
                                   kind='scatter', s=6, size=5, joint_kws=dict(color=colors[inv]))
                     .set_axis_labels(params['xlabel'], params['ylabel']))
                g.savefig(os.path.join(tmpdir, 'tmpimg.png'), bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close(g.fig)

                # set up axes in which to load saved image. remove axes spines and tick labels
                # so that y_label is still visible
                ax = plt.subplot(gs[feat_idx, 0])
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)
                ax.set_ylabel(params['title'], fontsize=8)

                # read image from file and show in axes
                img = misc.imread(os.path.join(tmpdir, 'tmpimg.png'))
                ax.imshow(img, aspect='equal', interpolation='none')

            plt.tight_layout(h_pad=0.05, w_pad=0.05)
            fig.savefig(os.path.join(tmpdir, fname+'.pdf'), dpi=600)
            plt.close(fig)

        # merge the pdf pages into a single pdf
        pdfs = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.pdf')]
        merger = PdfFileMerger()
        for pdf in pdfs:
            merger.append(FileIO(pdf, 'rb'))
        merger.write(outfile)


    ##################################################################################################
    # Generate PDF showing final clustering results for each ROI. Each ROI will be on a seperate page.
    ##################################################################################################
    outfile2d = os.path.join(BASE_OUTPUT_DIR, 'clusters_2d.pdf')
    outfile3d = os.path.join(BASE_OUTPUT_DIR, 'clusters_3d.pdf')

    print('Saving {} ...'.format(outfile2d))
    trackplot.roi_track_clusters(outfile2d, tracks['video'], tracks['coords'], track_labels,
                                 '2d', tracks['frame_num'])
    print('Saving {} ...'.format(outfile3d))
    trackplot.roi_track_clusters(outfile3d, tracks['video'], tracks['coords'], track_labels,
                                 '3d', tracks['frame_num'])

    quit()

    ###########################################
    # Show feature histograms for each cluster.
    ###########################################
    from matplotlib.backends.backend_pdf import PdfPages

    df = pd.DataFrame({'video': tracks['video'], 'length': tracks['length'], 'cluster': track_labels})
    params = {
        'length': {'range': (df.length.min(), df.length.max()), 'bins': 50}
    }

    with PdfPages(os.path.join(BASE_OUTPUT_DIR, 'cluster_features.pdf')) as pdf:
        for roi, inds in zip(tqdm.tqdm(roi_unq), roi_unq_tracks):
            group_unq = np.sort(df.loc[inds, 'cluster'].unique())
            group_kwargs = {g: {'color': c} for g, c in zip(group_unq, sns.husl_palette(len(group_unq)))}
            group_kwargs[-1] = {'color': 'black', 'alpha': 0.2}

            # plot 3d tracklets
            fig = plt.figure(figsize=(8, 6))
            fig.suptitle(roi)
            ax1 = fig.add_axes([0, 0, 1, 1], projection='3d')

            trackplot.tracklet_3d(tracks['coords'][inds], tracks['frame_num'][inds], ax=ax1,
                                  groups=df.loc[inds, 'cluster'].values,
                                  group_kwargs=group_kwargs)
            pdf.savefig()
            plt.close(fig)

            # plot features for each cluster. clusters are on rows and features on columns.
            fig, axes = plt.subplots(len(group_unq), 2, squeeze=False,
                                     gridspec_kw=dict(left=0.01, bottom=0.01, right=0.99, top=0.99))
            axes[0, 0].set_title('Length')
            axes[0, 1].set_title('Average velocity')

            for row in range(len(group_unq)):
                axes[row, 0].set_ylabel('cluster = {}'.format(group_unq[row]))
                axes[row, 0].hist(
                    df.loc[(df['video'] == roi) & (df['cluster'] == group_unq[row]), 'length'],
                    normed=True, color=group_kwargs[group_unq[row]]['color'],
                    **params['length']
                )
                axes[row, 0].set_ylim(0, 0.5)

                # axes[row, 1].plot(track_features['average_velocity'][],
                #                   'o', c=group_kwargs[group_unq[row]]['color'])

            pdf.savefig()
            plt.close(fig)

    quit()
    #####################################################################################
    # Split rois into training and test sets and match the clusters in the test set to
    # the most similar cluster in the training set.
    #####################################################################################
    from sklearn.model_selection import train_test_split

    rois_train, rois_test, _, _ = train_test_split(rois, [None]*len(rois), test_size=25)

    labels_train = set(track_labels[np.in1d(tracks['video'], rois_train['video'])]) - {-1}
    labels_test = set(track_labels[np.in1d(tracks['video'], rois_test['video'])]) - {-1}

    labels_test_match = dict()

    for l_test in tqdm.tqdm(labels_test, desc='finding similar clusters'):
        min_dist = np.inf
        train_label_match = None

        for l_train in labels_train:
            X1 = tracks['trajectory'][track_labels == l_test]
            X2 = tracks['trajectory'][track_labels == l_train]
            dist = trackdist.euclidean_sum(X1, X2)

            dist_min = np.amin(dist)
            if dist_min < min_dist:
                min_dist = dist_min
                train_label_match = l_train

        labels_test_match[l_test] = dict(min_dist=min_dist, train_label=train_label_match,
                                         train_roi=tracks['video'][track_labels == train_label_match][0])


    with tempfile.TemporaryDirectory() as tmpdir:
        for test_roiname in rois_test['video']:
            roi_labels = set(track_labels[tracks['video'] == test_roiname]) - {-1}  # cluster labels in roi
            if len(roi_labels) < 1:
                continue

            fig, ax = plt.subplots(nrows=2, ncols=len(roi_labels), figsize=(10, 8),
                                   subplot_kw={'projection': '3d'}, tight_layout=True, squeeze=False)
            fig.suptitle(test_roiname)
            ax[0, 0].set_ylabel('Test clusters')
            ax[1, 0].set_ylabel('Train clusters')

            for col, test_label in enumerate(roi_labels):
                ax[0, col].set_title('dist: {:.5f}'.format(labels_test_match[test_label]['min_dist']))
                trackplot.tracklet_3d(tracks['coords'][track_labels == test_label],
                                      tracks['frame_num'][track_labels == test_label],
                                      groups=[test_label]*np.count_nonzero(track_labels == test_label),
                                      ax=ax[0, col])
                train_label = labels_test_match[test_label]['train_label']
                trackplot.tracklet_3d(tracks['coords'][track_labels == train_label],
                                      tracks['frame_num'][track_labels == train_label],
                                      groups=[train_label]*np.count_nonzero(track_labels == train_label),
                                      ax=ax[1, col])
                ax[1, col].set_title(labels_test_match[test_label]['train_roi'])

            fig.savefig(os.path.join(tmpdir, os.path.splitext(os.path.basename(test_roiname))[0]+'.pdf'))
            plt.close(fig)

        # merge the pdf pages into a single pdf
        pdfs = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.pdf')]
        merger = PdfFileMerger()
        for pdf in pdfs:
            merger.append(FileIO(pdf, 'rb'))
        merger.write(os.path.join(BASE_OUTPUT_DIR, 'matched_clusters.pdf'))
