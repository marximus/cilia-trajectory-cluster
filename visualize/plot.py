import os
import tempfile
import multiprocessing

import matplotlib as mpl
# mpl.use('Agg')
mpl.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import scipy.cluster.hierarchy as hierarchy

import cilia.trajectory.cluster as cluster

mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 'xx-small'
mpl.rcParams['figure.titlesize'] = 12
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8


def video_grid(videos, correct, outfile, titles=None, n_cols=4, max_frames=None, cmap="RdYlGn", cmap_range=(0, 1), fps=10):
    """Show grid of videos with the background colored according to the 0/1 label.

    Parameters
    ----------
    videos : list of arrays, each array has shape (n_frames, height, width)
        Video arrays.
    corr : list of float, shape (n_videos,) in range [0, 1]
        The percentage of correct classifications for the corresponding video. The
        border of the videos will be colored according to these values.
    outfile : str
        Name of output file. Should have a .mp4 extension.
    n_cols : int
        Number of columns in grid.
    max_frames : int, optional
        Maximum number of frames to show. If None, all frames will be shown.
    cmap : str
        Colormap used to color backgrounds.
    cmap_range : tuple, (2,)
        The (min, max) range used when mapping values in `corr` to colors.
    """
    if len(videos) < 1:
        raise ValueError('videos is empty')
    assert len(videos) == len(correct)

    max_frames = int(np.amax([len(vid) for vid in videos])) if max_frames is None else max_frames
    n_rows = int(np.ceil(len(videos) / n_cols))
    colors = _values2colors(correct, cmap, cmap_range[0], cmap_range[1], clip=True)

    fig, axes = plt.subplots(n_rows, n_cols)

    ims = []
    for i, ax in enumerate(axes.flat):
        if i < len(videos):
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            # Increase border width and set color.
            for spine in ax.spines.values():
                spine.set_linewidth(3)
                spine.set_color(colors[i])
            # Set title of axes
            if titles is None:
                ax.set_title('corr={:.2f}'.format(correct[i]), fontdict={'fontsize': 6})
            else:
                ax.set_title('{} (corr={:.2f})'.format(titles[i], correct[i]), fontdict={'fontsize': 6})
            # Plot initial image into axes
            im = ax.imshow(videos[i][0],  cmap='gray', vmin=0, vmax=255, animated=True, aspect='equal')
            ims.append(im)
        else:
            ax.set_axis_off()

    def animate(idx):
        print(idx)
        for vid, im in zip(videos, ims):
            im.set_array(vid[idx])

    ani = animation.FuncAnimation(fig, animate, frames=max_frames)
    ani.save(outfile, fps=fps, bitrate=4000, dpi=200)
    plt.close(fig)


def _values2colors(values, cmap, vmin, vmax, clip):
    norm = mpl.colors.Normalize(vmin, vmax, clip)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    return [mpl.colors.rgb2hex(mapper.to_rgba(val)) for val in values]


def rois_img(img, polygons, fname, dpi=200):
    """Create and image showing all ROIs in a video.

    Parameters
    ----------
    img : array, shape (height, width)
        The grayscale image the polygons will be plotted over.
    polygons : 3D list
        3-D structure where each entry is a polygon, a (n, 2) array with (x, y)
        coordinates.
    fname : str
        Output file name. Should have a .png extension.
    """
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_axis_off()
    # Plot image
    ax.imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=255, aspect='equal')
    ax.autoscale(False)
    # Plot polygon lines
    coll = PolyCollection(polygons, facecolors='none', edgecolors='blue')
    ax.add_collection(coll)
    # Plot polygon points and annotate each point with its (x, y) coordinates
    for poly in polygons:
        poly = np.asarray(poly)
        xs, ys = poly[:, 0], poly[:, 1]
        ax.plot(xs, ys, '.r')
        for x, y in zip(xs, ys):
            ax.annotate('({}, {})'.format(x, y), xy=(x, y), textcoords='data', size=6)

    fig.savefig(fname, dpi=dpi)
    # plt.show()
    plt.close()


def _compute_fig_size(K, scale=(0.25, 0.1)):
    height, width = np.array(K.shape, dtype=float) * np.array(scale)[::-1]
    figsize = np.ceil((width, height))
    print('height={}, width={}'.format(height, width))
    return figsize


def matrix(df, filename, row_label_colors=None, col_label_colors=None, annot=True, fontsize=4):
    """Save colored image of 2d matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Matrix to save.
    filename : str
        Path to save image.
    row_label_colors : array_like, optional
        The colors of the labels on the vertical axis.
    col_label_colors : array_like, optional
        The colors of the labels on the horizontal axis.

    Returns
    -------
    None
    """
    if row_label_colors is not None: assert len(row_label_colors) == df.shape[0]
    if col_label_colors is not None: assert len(col_label_colors) == df.shape[1]

    # fig = plt.figure(figsize=_compute_fig_size(df))
    fig = plt.figure(figsize=(12, 50))
    ax = fig.add_axes([0, 0, 1, 1])

    sns.heatmap(df, ax=ax, cbar=False, annot=annot,
                annot_kws={'fontsize':fontsize} if annot else None)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=fontsize)

    # Color the row/column labels. The tick labels are returned from bottom to top
    # so we must flip them. We make sure the text of the labels matches that from the
    # pandas.DataFrame just to prevent a potential bug.
    if row_label_colors is not None:
        ticklabels = ax.get_yticklabels()[::-1]
        np.testing.assert_array_equal(df.index.values, np.array([tl.get_text() for tl in ticklabels], dtype=np.int))
        for label, color in zip(ticklabels, row_label_colors):
            label.set_color(color)

    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def save_confusion_matrices(patients, rois, tracks, fname, normalize=False):
    """
    Output confusion matrix for trajectories, ROIs and patients
    """
    y_trues = (patients.y_true.values, rois.y_true.values, tracks.y_true.values)
    y_preds = (patients.y_pred.values, rois.y_pred.values, tracks.y_pred.values)
    titles = ('Patients', 'ROIs', 'Trajectories')
    assert len(y_trues) == len(y_preds) == len(titles)
    ncols = len(y_trues)

    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(10, 4))

    for i, (y_true, y_pred, title) in enumerate(zip(y_trues, y_preds, titles)):
        labels = [{-1: 'N/A', 0: 'Normal', 1: 'Abnormal'}[y] for y in np.union1d(y_true, y_pred)]

        cmat = confusion_matrix(y_true, y_pred)
        if normalize:
            cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]

        df = pd.DataFrame(cmat, index=labels, columns=labels)

        cax = make_axes_locatable(axes[i]).append_axes('right', size='5%', pad=0.05)
        sns.heatmap(df, ax=axes[i], cbar_ax=cax, annot=True, fmt='d',
                    square=True, linewidths=1, linecolor='black',)
                    # annot_kws={'fontsize': 8})
        axes[i].set_title(title)
        axes[i].set_xlabel('Predicted label')
        axes[i].set_ylabel('True label')
        # plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize)
        # plt.setp(ax.get_xticklabels(), rotation=45, fontsize=fontsize)

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.6)
    fig.savefig(fname)
    plt.close(fig)


def roi_track_clusters(outfile, roinames, coords, clusters, end_frames=None, proj='3d'):
    """
    Create pdf file showing the clusters per ROI.

    Parameters
    ----------
    outfile : str
        File path of output pdf file.
    roinames
    coords
    clusters

    Returns
    -------

    """
    from PyPDF2 import PdfFileMerger
    from io import FileIO

    if not len(roinames) == len(coords) == len(clusters):
        raise ValueError('rois, coords, and clusters must have same size')
    if proj not in ('2d', '3d'):
        raise ValueError("proj must be '2d' or '3d'")
    if proj == '3d' and end_frames is None:
        raise ValueError("end_frames parameter must be used when proj == '3d'")

    outfile = outfile if outfile.endswith('.pdf') else outfile + '.pdf'

    rois_uniq, unq_inv, unq_cnt = np.unique(roinames, return_counts=True, return_inverse=True)
    indices = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))

    # save pdf of each ROI separately in a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, (roiname, ind) in enumerate(zip(rois_uniq, indices)):
            # mask contains indices of trajectories in roi where the cluster != -1
            mask = ind[clusters[ind] != -1]
            n_clusters = len(np.unique(mask))

            if n_clusters > 0:
                fname = os.path.join(tmp_dir, os.path.splitext(os.path.basename(roiname))[0] + '.pdf')
                if proj == '2d':
                    track_clusters_2d(fname, coords[mask], clusters[mask], 2, 5, title=roiname)
                elif proj == '3d':
                    track_clusters_3d(fname, coords[mask], end_frames[mask], clusters[mask], title=roiname)

        # merge all saved pdfs into a single pdf
        # http://stackoverflow.com/questions/3444645/merge-pdf-files
        pdfs = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
        merger = PdfFileMerger()
        for pdf in pdfs:
            merger.append(FileIO(pdf, 'rb'))
        merger.write(outfile)


def _squeeze(n_rows, n_cols, n):
    """
    Reduce grid with size n_rows x n_cols to fit n items, if possible.
    Return new n_rows and n_cols.
    """
    # row and column index of last element in grid
    r = (n-1) // n_cols
    c = (n-1) % n_cols
    # remove unused rows
    if r < n_rows-1:
        n_rows = r+1
    # remove unused columns only if there is a single row
    if n_rows == 1:
        n_cols = c+1

    return n_rows, n_cols


def track_clusters_2d(outfile, coords, labels, n_rows=2, n_cols=5, squeeze=False, title=None):
    """
    Show clusters of trajectories.

    Parameters
    ----------
    coords : array, shape (n_tracks, track_length, 2)
        The x and y coordinates of the trajectories.
    labels : array, shape (n_tracks,)
        Cluster label for each trajectory. The label -1 is used when a
        trajectory does not belong to any cluster.

    Returns
    -------
    None
    """
    if len(coords) != len(labels):
        raise ValueError('number of trajectories must match number of labels')

    labels_uniq = np.unique(labels)
    colors = sns.husl_palette(len(labels_uniq))
    n_clusters = len(labels_uniq)

    if squeeze:
        # row and column index of last element in grid
        r, c = (n_clusters-1) // n_cols, (n_clusters-1) % n_cols
        # remove unused rows
        if r < n_rows-1:
            n_rows = r+1
        # remove unused columns only if there is a single row
        if n_rows == 1:
            n_cols = c+1
        print('len={}, n_rows={}, n_cols={}'.format(n_clusters, n_rows, n_cols))

    n_pages = int(np.ceil(len(labels_uniq)/(n_rows*n_cols)))
    with PdfPages(outfile) as pdf:
        for p in range(n_pages):
            i = p*n_rows*n_cols
            j = p*n_rows*n_cols + n_rows*n_cols

            # set up figure and grid of axes
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 9), tight_layout=True)
            axes = np.atleast_2d(axes)
            # hide axes that will not be used
            for idx, ax in enumerate(axes.flat):
                if idx >= len(labels_uniq[i:j]):
                    ax.axis('off')
            # show title on first page
            if p == 0 and title:
                fig.suptitle(title)

            for ax, label, c in zip(axes.flat, labels_uniq[i:j], colors[i:j]):
                xys = coords[labels == label]

                # set axis limits
                pad = 0.5
                ax.set_xlim(xys[:, :, 0].min()-pad, xys[:, :, 0].max()+pad)
                ax.set_ylim(xys[:, :, 1].min()-pad, xys[:, :, 1].max()+pad)
                ax.set_title('cluster={}'.format(label))
                ax.set_aspect('equal')

                # plot trajectories in cluster
                for xy in xys:
                    ax.plot(xy[:, 0], xy[:, 1], '-o', lw=0.7, ms=2.0, color=c)

                # legend with color and # of trajectories in cluster
                ax.legend(handles=[Patch(color=c, label=str(len(xys)))], loc='best')

            pdf.savefig()
            plt.close(fig)


def track_clusters_3d(outfile, coords, end_frames, labels, title=None):
    """
    Show clusters of trajectories in 3d.

    Parameters
    ----------
    outfile : str
    coords : array, shape (n_tracks, track_length, 2)
        The x and y coordinates of the trajectories.
    end_frames :
    labels : array, shape (n_tracks,)
        Cluster label for each trajectory. The label -1 is used when a
        trajectory does not belong to any cluster.

    Returns
    -------
    None
    """
    if not (len(coords) == len(labels) == len(end_frames)):
        raise ValueError('number of trajectories must match number of labels')

    labels_uniq, labels_cnt = np.unique(labels, return_counts=True)
    colors = sns.hls_palette(len(labels_uniq))

    with PdfPages(outfile) as pdf:
        # set up figure and 3d axis
        fig, ax = plt.subplots(tight_layout=True, figsize=(16, 9),
                               subplot_kw={'projection': '3d'})
        if title:
            ax.set_title(title)

        # plot trajectories in each cluster
        for l, c in zip(labels_uniq, colors):
            indices = np.flatnonzero(labels == l)
            for idx, (xy, end) in enumerate(zip(coords[indices], end_frames[indices])):
                ax.plot(xy[:, 0], xy[:, 1], np.arange(end+1-len(xy), end+1), '-o',
                        zdir='y', lw=0.7, ms=1.5, color=c)

        # show legend
        ax.legend(handles=[Patch(color=c, label=str(cnt)) for c, cnt in zip(colors, labels_cnt)])

        # mimic video frames
        xy = (coords[:, :, 0].min(), coords[:, :, 1].min())
        w = coords[:, :, 0].max() - coords[:, :, 0].min()
        h = coords[:, :, 1].max() - coords[:, :, 1].min()
        for z in range(0, end_frames.max(), 50):
            rect = Rectangle(xy, w, h, fill=False, color='black',
                             alpha=0.3, lw=0.3)
            ax.add_patch(rect)
            art3d.pathpatch_2d_to_3d(rect, z=z, zdir='y')

        # set axis limits
        pad = 0.5
        ax.set_xlim3d(coords[:, :, 0].min()-pad, coords[:, :, 0].max()+pad)
        ax.set_ylim3d(0, end_frames.max()+1)
        ax.set_zlim3d(coords[:, :, 1].min()-pad, coords[:, :, 1].max()+pad)

        ax.set_xlabel('Video width')
        ax.set_ylabel('Video frames')
        ax.set_zlabel('Video height')

        ax.view_init(elev=20, azim=12)
        _set_axes_equal(ax)

        pdf.savefig()
        plt.close(fig)


# Taken from
# http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def _set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    # y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    # y_range = abs(y_limits[1] - y_limits[0])
    # y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    # plot_radius = 0.5*max([x_range, y_range, z_range])
    plot_radius = 0.5*max([x_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    # ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def dendrogram(outfile, Z, max_d=None):
    fig = plt.figure(figsize=(50, 15), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.set_xlabel('initial clusters of trajectories')
    ax.set_ylabel('distance')

    # hierarchy.dendrogram(Z, 200, 'level', no_labels=True, color_threshold=0.2, ax=ax)
    hierarchy.dendrogram(Z, ax=ax, color_threshold=max_d)

    if max_d:
        ax.axhline(y=max_d, c='k')

    fig.savefig(outfile)
    plt.close(fig)


def show_merged_clusters(track_coords, clust1, clust2, outfile,
                         n_rows=2, n_cols=5):
    """

    Parameters
    ----------
    track_coords : ndarray, shape (n_tracks, track_length, 2)
        Coordinates of dense trajectories.
    clusters : seq of ndarray
        Indices of dense trajectories in each cluster.
    merged_clust : ndarray, shape (len(clusters),)
        Indices of clusters in each merged cluster.
    outfile : str
        Path of output file.

    Returns
    -------
    None
    """
    assert len(track_coords) == len(clust1) == len(clust2)
    #
    # outfile = outfile if outfile.endswith('.pdf') else outfile + '.pdf'
    # # n_cols = int(np.sqrt(len(merged_clust_indices[i])))
    # # n_rows = int(np.ceil(len(merged_clust_indices[i]) / n_cols))
    #
    # with PdfPages(outfile) as pdf:
    #     # loop over merged clusters
    #     for i in
    #         # show each cluster (of trajectories) in a different color
    #         colors = sns.husl_palette(len(merged_clust_indices[i]))
    #
    #         # loop over child clusters in chunks
    #         for j in range(0, len(merged_clust_indices[i]), n_rows*n_cols):
    #             inds = merged_clust_indices[i][j:j+n_rows*n_cols]
    #             n_clusts = len(inds)
    #
    #             # reduce dims of grid if there are unnecessary columns/rows
    #             rows, cols = _squeeze(n_rows, n_cols, n_clusts)
    #
    #             # set up figure and grid of axes
    #             fig, axes = plt.subplots(rows, cols, tight_layout=True)
    #             axes = np.atleast_2d(axes)
    #             if j == 0:
    #                 fig.suptitle('merged cluster {}, n_clusters={}'.format(i, len(merged_clust_indices[i])))
    #             for ax in axes.flat:
    #                 ax.set_xticklabels([])
    #                 ax.set_yticklabels([])
    #                 ax.set_aspect('equal')
    #
    #             # loop over clusters (of trajectories)
    #             for k in range(n_clusts):
    #                 print('j+k={}'.format(j+k))
    #                 axes.flat[k].set_title('cluster {}'.format(j+k))
    #                 # loop over trajectories in cluster
    #                 for xy in track_coords[clusters[inds[j+k]]]:
    #                     axes.flat[k].plot(*xy.T, '-o', c=colors[j+k],
    #                                       ms=2.0, lw=0.5)
    #
    #             pdf.savefig(fig)
    #             plt.close(fig)
