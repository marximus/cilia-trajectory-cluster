import numpy as np


def curvature(xs, ys):
    """
    Curvature for each point on a parametric curve.

    Curvature will be zero for a line and larger the more it deviates from a line.

    Parameters
    ----------
    xs : ndarray, shape (n_curves, n_pts)
        X coordinates.
    ys : ndarray, shape (n_curves, n_pts)
        Y coordinates.

    Returns
    -------
    k : ndarray, shape (n_curves, n_pts)
        Curvature for each point

    See Also
    --------
    https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/curvature
    """
    x_prime = np.gradient(xs, axis=1, edge_order=2)
    y_prime = np.gradient(ys, axis=1, edge_order=2)
    xy_prime_mag = np.linalg.norm(np.dstack((x_prime, y_prime)), axis=2)

    Tx = x_prime / xy_prime_mag
    Ty = y_prime / xy_prime_mag

    Tx_prime = np.gradient(Tx, axis=1, edge_order=2)
    Ty_prime = np.gradient(Ty, axis=1, edge_order=2)
    Txy_prime_mag = np.linalg.norm(np.dstack((Tx_prime, Ty_prime)), axis=2)

    # curvature of each point of each curve
    k = Txy_prime_mag / xy_prime_mag

    return k


#TODO: implement fps arguments
def velocity(lines, fps=None):
    """
    Compute velocity between pairs of points. 
    
    Parameters
    ----------
    lines : ndarray, shape (n_lines, n_pts, 2)
        Lines for which to compute the velocity.
         
    Returns
    -------
    v : ndarray, shape (n_lines, n_pts, 2)
        x and y components of velocity between each pair of points.
        
    References
    ----------
    http://mathworld.wolfram.com/VelocityVector.html
    """
    x_prime = []
    y_prime = []
    for i in range(len(lines)):
        x_prime.append(np.gradient(lines[i, :, 0], edge_order=2))
        y_prime.append(np.gradient(lines[i, :, 1], edge_order=2))
    x_prime = np.array(x_prime)
    y_prime = np.array(y_prime)

    if fps is not None:
        pass

    return np.rollaxis(np.array((x_prime, y_prime)), axis=0, start=3)


def average_velocity(coords):
    """
    Average velocity for points on a parametric curve.

    Helps seperate tracklets moving at varying pace.

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    Anjum, Nadeem, and Andrea Cavallaro. "Multifeature object trajectory clustering for video analysis." IEEE
    Transactions on Circuits and Systems for Video Technology 18.11 (2008): 1555-1564.
    """
    n_pts = coords.shape[1]
    return (1/(n_pts-1)) * np.sum(coords[:, 1:, :] - coords[:, :-1, :], axis=1)


def directional_distance(coords):
    """Directional distance between first and last points in tracklet.

    Encodes the direction of motion and helps distinguish longer tracklets from shorter ones
    and also tracklets in opposite directions.

    Parameters
    ----------
    coords

    Returns
    -------

    References
    ----------
    Anjum, Nadeem, and Andrea Cavallaro. "Multifeature object trajectory clustering for video analysis." IEEE
    Transactions on Circuits and Systems for Video Technology 18.11 (2008): 1555-1564.
    """
    return coords[:, -1, :] - coords[:, 0, :]


def trajectory_mean(coords):
    """Trajectory mean.

    Helps distinguish tracklets belonging to different regions on the image plane.

    Parameters
    ----------
    coords

    Returns
    -------

    References
    ----------
    Anjum, Nadeem, and Andrea Cavallaro. "Multifeature object trajectory clustering for video analysis." IEEE
    Transactions on Circuits and Systems for Video Technology 18.11 (2008): 1555-1564.
    """
    n_pts = coords.shape[1]
    return (1/n_pts) * coords.sum(axis=1)


def directional_histogram(coords, n_bins):
    """
    Trajectory directional histogram.

    Parameters
    ----------
    coords
    n_bins : int
        Number of bins. The interval [-pi,pi) will be divided into n_bins
        equal subintervals. The length of each subinterval is 2*pi/n_bins.

    Returns
    -------
    hist

    References
    ----------
    Anjum, Nadeem, and Andrea Cavallaro. "Multifeature object trajectory clustering for video analysis." IEEE
    Transactions on Circuits and Systems for Video Technology 18.11 (2008): 1555-1564.

    "A Coarse-to-Fine Strategy for Vehicle Motion Trajectory Clustering", Li, Xi et al.
    """
    x = coords[:, :, 0]
    y = coords[:, :, 1]

    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]

    # angles = np.arctan2(dy, dx) * 180 / np.pi
    # angles[angles<0] = angles[angles<0] + 360.0
    angles = np.arctan2(dy, dx)

    bins = np.linspace(-np.pi, np.pi, n_bins+1, endpoint=True)
    hist = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, angles)
    # normalize
    total_pts = angles.shape[1]
    hist = hist / total_pts

    return hist, bins


# def curvature2(xs, ys):
#     """
#     Sum of curvatures for each point on a parametric curve.
#
#     Curvature will be zero for a line and larger the more it deviates from a line.
#
#     Parameters
#     ----------
#     xs : ndarray, shape (n_curves, n_pts)
#         X coordinates.
#     ys : ndarray, shape (n_curves, n_pts)
#         Y coordinates.
#
#     Returns
#     -------
#     S : ndarray, shape (n_curves,)
#         Sum of curvature values for each point.
#
#     See Also
#     --------
#     https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/curvature
#     """
#     dx = np.gradient(xs)
#     ddx = np.gradient(dx)
#     dy = np.gradient(ys)
#     ddy = np.gradient(dy)
#
#     num = dx * ddy - dy * ddx
#     denom = (dx * dx + dy * dy) ** (3 / 2)
#
#     k = num / denom
#
#     # sum curvatures of points for each curve
#     S = np.sum(k, axis=1)
#
#     return S

##################################################
# Normalize feature spaces of each ROI separately
##################################################
# track_features_norm = {name: np.empty_like(track_features[name]) for name in track_features}
# for name, arr in track_features.items():
#     for roi, track_inds in zip(roi_unq, roi_unq_tracks):
#         track_features_norm[name][track_inds] = preprocessing.scale(track_features[name][track_inds])


##############################################################
# Visualize feature space vs normed feature space for each ROI
##############################################################
# outfile = os.path.join(BASE_OUTPUT_DIR, 'trajectory_features_normalized.pdf')
# print('Saving {} ...'.format(outfile))
#
# with tempfile.TemporaryDirectory() as tmpdir:
#     for roi, track_inds in zip(roi_unq, roi_unq_tracks):
#         # there is a problem with numpy arrays containing single elements so for now just duplicate the
#         # element to create an array of length 2 if a length 1 array is present
#         if len(track_inds) == 1:
#             track_inds = np.repeat(track_inds, 2)
#
#         fname = os.path.splitext(os.path.basename(roi))[0]+'.pdf'
#         print('Saving {}'.format(fname))
#
#         fig, axes = plt.subplots(nrows=2, ncols=len(track_features))
#         fig.suptitle(fname)
#
#         # fill first row with features
#         for feat_idx, name in enumerate(sorted(track_features.keys())):
#             # save plot as image to file
#             # see http://stackoverflow.com/questions/37945495/python-matplotlib-save-as-tiff on how to save to memory rather than a file
#             g = sns.jointplot(track_features[name][track_inds, 0],
#                               track_features[name][track_inds, 1],
#                                kind='scatter', s=6, size=5)
#             g.set_axis_labels(feature_plot[name]['xlabel'], feature_plot[name]['ylabel'])
#             g.savefig(os.path.join(tmpdir, 'tmpimg.png'), bbox_inches='tight', pad_inches=0, dpi=300)
#             plt.close(g.fig)
#
#             # read image from file and show in axes
#             img = misc.imread(os.path.join(tmpdir, 'tmpimg.png'))
#             axes[0, feat_idx].axis('off')
#             axes[0, feat_idx].imshow(img, aspect='equal', interpolation='none')
#             axes[0, feat_idx].set_title(feature_plot[name]['title'], fontsize=6)
#
#         # fill second row with normalized features
#         for feat_idx, name in enumerate(sorted(track_features_norm.keys())):
#             # save plot as image to file
#             # see http://stackoverflow.com/questions/37945495/python-matplotlib-save-as-tiff on how to save to memory rather than a file
#             g = sns.jointplot(track_features_norm[name][track_inds, 0],
#                               track_features_norm[name][track_inds, 1],
#                                kind='scatter', s=6, size=5)
#             g.set_axis_labels(feature_plot[name]['xlabel'], feature_plot[name]['ylabel'])
#             g.savefig(os.path.join(tmpdir, 'tmpimg.png'), bbox_inches='tight', pad_inches=0, dpi=300)
#             plt.close(g.fig)
#
#             # read image from file and show in axes
#             img = misc.imread(os.path.join(tmpdir, 'tmpimg.png'))
#             axes[1, feat_idx].axis('off')
#             axes[1, feat_idx].imshow(img, aspect='equal', interpolation='none')
#             axes[1, feat_idx].set_title('Normal '+feature_plot[name]['title'], fontsize=6)
#
#         fig.savefig(os.path.join(tmpdir, fname))
#         plt.close(fig)
#
#     # merge the pdf pages into a single pdf
#     pdfs = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.pdf')]
#     merger = PdfFileMerger()
#     for pdf in pdfs:
#         merger.append(FileIO(pdf, 'rb'))
#     merger.write(outfile)



####################################################################################
# Removing straight trajectories
####################################################################################
# compute curvature for each trajectory, removing any lines where curvature is nan/inifinite
# mask = np.all(np.isfinite(curvature), axis=1)
# curve = curve[mask]
# roinames = roinames[mask]

# threshold curvature to remove straight trajectories
# mask = np.any(curve < curve_threshold, axis=1)
# n_bad_tracks = len(np.flatnonzero(mask))
# print('# of straight trajectories removed: {} ({:.2%})'.format(n_bad_tracks, n_bad_tracks/len(tracks)))
# rois_uniq, unq_inv, unq_cnt = np.unique(tracks['video'], return_counts=True, return_inverse=True)
# tr_indices = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
#
# # save pdf of all ROIs and the removed straight trajectories
# labels = np.zeros(len(tracks), dtype=np.int)
# labels[mask] = 1
# for roiname, ind in zip(rois_uniq, tr_indices):
#     fname = os.path.splitext(os.path.basename(roiname))[0] + '.pdf'
#     print('Saving {}'.format(fname))
#     plot.trajectory_3d(tracks['coords'][ind], tracks['frame_num'][ind],
#                        groups=labels[ind],
#                        save_path=os.path.join(BASE_OUTPUT_DIR, fname))

#     trs = tracks[curve_mask]
#     labels = np.zeros(len(trs), np.int)
#     print('Saving pruned tracks (2d and 3d) ...')
# plot.roi_track_clusters(os.path.join(BASE_OUTPUT_DIR, 'pruned_tracks_3d.pdf'),
#                         trs['video'], trs['coords'], labels, '3d', trs['frame_num'], progress=True)
# plot.roi_track_clusters(os.path.join(BASE_OUTPUT_DIR, 'pruned_tracks_2d.pdf'),
#                         trs['video'], trs['coords'], labels, '2d', progress=True)


