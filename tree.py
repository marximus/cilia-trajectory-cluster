import os
import tempfile
import multiprocessing
import subprocess

import ete3
from scipy.cluster import hierarchy
from scipy.misc import imread
import numpy as np
import ffmpy

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

FFMPEG_PATH = "/home/mam588/miniconda3/envs/cilia/bin/ffmpeg"


def from_ClusterNode(root):
    """
    Converts a scipy.cluster.hierarchy.ClusterNode object into an ETE Tree object.

    Parameters
    ----------
    root : scipy.cluster.hierarchy.ClusterNode instance
        ClusterNode instance to convert to ETE Tree object.

    Returns
    -------
    tree : ete3.Tree instance
        New tree.
    """
    if root is None:
        return None

    # create copy of root node
    ete3_node = ete3.Tree(name=root.get_id(), dist=root.dist)

    # recursively create clone of left and right sub tree
    if root.get_left():
        new_node = from_ClusterNode(root.get_left())
        ete3_node.add_child(new_node)
    if root.get_right():
        new_node = from_ClusterNode(root.get_right())
        ete3_node.add_child(new_node)

    return ete3_node


def vis_cluster_tree(outfile, root, tracks, max_frames=None, fps=10, debug=False):
    """
    Visualize merged trajectory clusters.

    The leaf node ids are the cluster ids of the original clusters. The leaf node ids
    must match the keys in `tracks`, which maps the original cluster ids to the trajectories
    in the original cluster.

    Parameters
    ----------
    outfile : str
        Output file name.
    root : scipy.cluster.hierarchy.ClusterNode instance
        Root of subtree extracted from hierarchical clustering tree.
    tracks : dict
        Mapping from original cluster ids to trajectories in cluster.
    max_frames : int
        Maximum number of frames to show.
    fps : int
        Frame rate of output video.
    debug : bool
        If True, show stdout and stderr of ffmpeg in running shell. If False, stdout and stderr
        will be redirected to os.devnull.

    Returns
    -------
    None
    """
    # convert tree from scipy.cluster.hierarchy.ClusterNode instance to ete3.Tree. Do this
    # so the ete3 library can be used to show the tree.
    ete3_tree = from_ClusterNode(root)

    # check that ids of leaf nodes in tree rooted at `root` match keys in `tracks`. These are
    # the ids of the original clusters that were merged.
    assert sorted(ete3_tree.get_leaf_names()) == sorted(tracks.keys())
    cluster_ids = np.array(list(tracks.keys()))
    n_clusters = len(cluster_ids)

    # map cluster ids to video. Since trajectory clusters were found in each ROI independently,
    # the trajectories of each cluster should have the same `video` attribute.
    videos = dict()
    videonames = dict()
    for c_id in cluster_ids:
        if not np.all(tracks[c_id]['video'] == tracks[c_id]['video'][0]):
            raise ValueError('all trajectories in cluster must originate from same video')
        videos[c_id] = np.load(tracks[c_id]['video'][0])
        videonames[c_id] = os.path.basename(tracks[c_id]['video'][0])

    # set number of frames to animate as the minimum between the
    # length of the shortest video and `max_frames`.
    n_frames = min(map(len, videos.values()))
    if max_frames:
        n_frames = min(n_frames, max_frames)

    # save sequences of images animating trajectories for each original cluster
    anim = dict()
    for c_id in cluster_ids:
        tempdir = tempfile.TemporaryDirectory()
        images = save_track_images(videos[c_id][:n_frames], tracks[c_id], tempdir.name)
        anim[c_id] = {'tempdir': tempdir, 'images': images}

    # temporary directory used to save frames of dendrogram animation
    dendogram_dir = tempfile.TemporaryDirectory()

    # save sequences of images animating dendrogram
    for frame_idx in range(n_frames):
        if frame_idx == 0:
            for node in ete3_tree.traverse():
                if node.is_leaf():
                    # original cluster id
                    attrface = ete3.faces.AttrFace('name')
                    attrface.rotation = 270
                    # image
                    imgface = ete3.faces.ImgFace(anim[node.name]['images'][frame_idx])
                    imgface.rotation = 270
                    node.add_feature('imgface', imgface)
                    # name of video
                    videoname_face = ete3.faces.TextFace(videonames[node.name], fsize=6)
                    videoname_face.rotation = 270
                    # add faces to leaf node
                    node.add_face(attrface, 0, 'branch-right')
                    node.add_face(imgface, 1, 'branch-right')
                    node.add_face(videoname_face, 2, 'branch-right')
                else:
                    distface = ete3.faces.AttrFace('dist', formatter="%0.3f")
                    node.add_face(distface, 0, 'branch-top')
        else:
            for node in ete3_tree.get_leaves():
                node.imgface.img_file = anim[node.name]['images'][frame_idx]

        # save dendogram image
        fname = os.path.join(dendogram_dir.name, 'img{0:05d}.png'.format(frame_idx))
        ts = ete3.TreeStyle()
        ts.show_leaf_name = False
        ts.show_scale = False
        # ts.show_branch_length = False
        ts.rotation = 90
        ete3_tree.render(fname, tree_style=ts)

    # check that images have same size
    sizes = [imread(os.path.join(dendogram_dir.name, f)).shape
             for f in os.listdir(dendogram_dir.name)]
    if not all(x==sizes[0] for x in sizes):
        quit('dendogram images have different sizes')

    # stitch together dendogram images
    # see https://trac.ffmpeg.org/wiki/Encode/H.264 for guide to encoding videos using x264 encoder
    # -preset options are ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow.
    # if no -preset option is given to FFMPEG, it will default to medium.
    ff = ffmpy.FFmpeg(
        FFMPEG_PATH,
        inputs={os.path.join(dendogram_dir.name, 'img%05d.png'): ('-framerate', str(fps))},
        outputs={outfile: ('-c:v', 'libx264',
                           '-pix_fmt', 'yuv420p',
                           '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                           '-preset', 'slower')},
        global_options=('-y',)       # -y overwrites file if exists
    )
    handle = None if debug else subprocess.DEVNULL
    ff.run(stderr=handle, stdout=handle)

    # remove temporary directories
    for k in anim.keys():
        anim[k]['tempdir'].cleanup()
    dendogram_dir.cleanup()

    return None


def save_track_images(video, tracks, outdir, width=150, height=150, color=None):
    """
    Save image sequence animating dense trajectories and return the
    paths of the saved images.

    Parameters
    ----------
    video : ndarray, shape (frames, height, width)
        Video from which the dense trajectories were computed.
    tracks : ndarray, shape (n_tracks, track_length, 2)
        Dense trajectories.
    outdir : str
        Path to directory where images will be saved.

    Returns
    -------
    images : seq of str
        Sequence of file paths to saved images.
    """
    images = []

    with multiprocessing.Pool() as pool:
        results = []
        for frame_idx in range(len(video)):
            # find trajectories in this frame
            trs = []
            for tr, end in zip(tracks['coords'], tracks['frame_num']):
                if _is_track_active(tr, end, frame_idx):
                    start_frame = end - len(tr) + 1
                    tr = tr[:frame_idx-start_frame+1]
                    trs.append(tr)

            # save image
            f = os.path.join(outdir, '{0:05d}.png'.format(frame_idx))
            # images.append(f)
            res = pool.apply_async(_save_track_image,
                                 (f, video[frame_idx], trs, width, height),
                                 {'color': color})
            results.append(res)

        for res in results:
            images.append(res.get())

    return images


def _save_track_image(outfile, img, tracks, width, height, color=None, title=None):
    """Save width x height image of dense trajectories"""
    dpi = 100
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_subplot(111, aspect='equal')
    if title:
        ax.set_title(title)

    # hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # turn off axis grid
    ax.grid(False)

    # show image
    ax.imshow(img, cmap='gray', interpolation='none', vmin=0, vmax=255, aspect='equal')
    ax.autoscale(False)

    # color trajectories blue if no color passed
    color = 'b' if color is None else color

    # show trajectories
    for tr in tracks:
        # show head of trajectory
        ax.plot(tr[-1, 0], tr[-1, 1], 'o', ms=1.2, c='r')
        # show body of trajectory
        if tr.shape[0] > 1:
            ax.plot(*tr.T, '-', lw=0.6, c=color)

    fig.savefig(outfile, dpi='figure', bbox_inches='tight')
    plt.close(fig)

    return outfile


def _is_track_active(track, end_frame, frame_num):
    """Return True if trajectory exists at given frame."""
    n_pts = len(track)
    start_frame = end_frame - n_pts + 1
    if start_frame < 0:
        quit('bad start_frame')
    pose_idx = frame_num - start_frame
    if pose_idx < 0:  # start_frame is larger than frame_num
        return False
    if pose_idx >= n_pts:  # trajectory already passed
        return False
    return True


