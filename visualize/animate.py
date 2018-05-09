from itertools import zip_longest
import time

import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = "/home/mam588/miniconda3/envs/cilia/bin/ffmpeg"
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

import numpy as np
# import seaborn as sns


def vector_curl(video, curl, flow, outfile=None, fps=10):
    """Animate curl.

    Parameters
    ----------
    video : str
        Path to .npy file of video frames.
    curl : str
        Path to .npy file of curl.
    """
    video = np.load(video)
    curl = np.load(curl)
    flow = np.load(flow)
    assert video.shape[1:] == curl.shape[1:] == flow.shape[2:]
    num_frames = np.amin((len(video), len(curl), flow.shape[1]))

    step = 5
    _, _, h, w = flow.shape
    u, v = flow

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    X, Y = X[::step, ::step], Y[::step, ::step]
    U, V = u[:, ::step, ::step], v[:, ::step, ::step]

    fig, ax = plt.subplots(tight_layout=True)
    im1 = ax.imshow(video[0], cmap='gray', interpolation='nearest', vmin=0, vmax=255, animated=True, aspect='equal')
    im2 = ax.imshow(curl[0], cmap='RdBu_r', interpolation='nearest', animated=True, aspect='equal', alpha=0.2)
    quiv = ax.quiver(X, Y, U[0], V[0], angles='xy', scale_units='xy', scale=1,
                     units='inches', width=0.008, headlength=2, headaxislength=2)

    def animate(idx):
        print(idx)
        im1.set_array(video[idx])
        im2.set_array(curl[idx])
        quiv.set_UVC(U[idx], V[idx])

    ani = animation.FuncAnimation(fig, animate, frames=range(num_frames-1))
    if outfile:
        print('Saving {}'.format(outfile))
        ani.save(outfile, fps=fps, dpi=400, bitrate=4000)
    else:
        plt.show()
    plt.close()


def tracks_grid(outfile, videos, tracks, n_rows, n_cols, fps=10):
    """
    Animate each video and its trajectories in a cell of a n_rows x n_cols grid.

    Parameters
    ----------
    outfile : str
        Name of output file.
    videos : seq of str
        Paths to .npy video files.
    tracks : seq of structured array
        Each item in seq is a structured array with the dense trajectories to plot for each video.
        The structured arrays must have fields 'coords' and 'end_frames'.
    n_rows : int
        Number of rows in grid.
    n_cols : int
        Number of columns in grid.

    Returns
    -------
    None

    See Also
    --------
    https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/
    """
    assert len(videos) == len(tracks)
    if len(videos) > n_rows*n_cols:
        raise ValueError('There are more videos than cells in the {} x {} grid'.format(n_rows, n_cols))

    # load videos from .npy files
    videos = [np.load(path) for path in videos]
    n_frames = max(map(len, videos))
    # n_frames = 25

    outfile = outfile if outfile.endswith('.mp4') else outfile+'.mp4'

    # set up figure and grid of axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.03, hspace=0.03)
    for ax in axes.flat:
        ax.axis('off')

    # set up images and the data structure where the artists of trajectories will be saved
    # when a trajectory is first plotted, it will create the necessary artists. When the
    # last frame of the trajectory is plotted it will delete the artist
    ims, points, lines = [], [], []
    for ax, vid, trs in zip(axes.flat, videos, tracks):
        im = ax.imshow(vid[0], cmap='gray', interpolation='nearest', vmin=0, vmax=255,
                       rasterized=True, aspect='equal', animated=True)
        ax.set_xlim(0, vid.shape[2])
        ax.set_ylim(0, vid.shape[1])

        ims.append(im)
        points.append([None]*len(trs))
        lines.append([None]*len(trs))

    # animation function. Called with frame number
    def animate(frame_idx):
        for ax, im, vid, trs, pts, lns in zip(axes.flat, ims, videos, tracks, points, lines):
            # update video frame
            if frame_idx < len(vid):
                im.set_array(vid[frame_idx])

            # plot trajectories in video
            for tr_idx, (coords, end) in enumerate(zip(trs['coords'], trs['frame_num'])):
                start = end-len(coords)+1

                # if we are done plotting a trajectory, delete it's artists
                if end+1 == frame_idx:
                    ax.lines.remove(pts[tr_idx])
                    ax.lines.remove(lns[tr_idx])
                # or create the artist(s) if they are needed
                if start == frame_idx:
                    pts[tr_idx] = ax.plot([], [], '+', c='r', ms=5.0, animated=True)[0]
                    lns[tr_idx] = ax.plot([], [], '-', c='g', lw=1.0, animated=True)[0]

                if start < frame_idx <= end:
                    xy = coords[:frame_idx-start+1]
                    # head of trajectory
                    pts[tr_idx].set_data([xy[-1, 0]], [xy[-1, 1]])
                    # lines connecting points
                    if len(xy) <= 1:
                        quit('Error')
                    if len(xy) > 1:
                        lns[tr_idx].set_data(*list(zip(*xy)))

    writer = animation.writers['ffmpeg'](fps=fps, bitrate=2000,
                                         # extra_args=['-pix_fmt', 'yuv420p'],
                                         extra_args=['-c:v', 'libx264', '-preset', 'medium'])

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, blit=False)
    anim.save(outfile, writer=writer, savefig_kwargs={'facecolor': '#efeaea'})

    plt.close(fig)


def show_tracks(video, tracks, end_frames, labels=None, title=None, fname=None, fps=5, dpi=300, bitrate=1000):
    """
    Animate dense trajectories.

    Parameters
    ----------
    video : array, shape (f, h, w)
        Video frames.
    tracks : array_like of 2-d arrays
        Each item is the (x, y) coordinates of a trajectory.
    end_frames : array_like of int
        The frame that each trajectory ends on.
    fps : int
        Frame rate of output.

    Returns
    -------
    None
    """
    if len(tracks) != len(end_frames):
        raise ValueError('length of tracks must equal length of end_frames')
    if labels is not None:
        uniq_labels = np.unique(labels)
        colors = dict(zip(uniq_labels, sns.color_palette('dark', n_colors=len(uniq_labels))))
        colors[-1] = 'k'
        colors = [colors[l] for l in labels]
    else:
        colors = np.full(shape=len(tracks), fill_value="b", dtype=object)

    # create array with shape (num_tracks, num_frames, 2)
    arr = np.full(shape=(len(tracks), len(video), 2), fill_value=np.nan)
    arr_mask = np.full(shape=(len(tracks), len(video)), fill_value=False, dtype=np.bool)
    for i, (track, end_frame) in enumerate(zip(tracks, end_frames)):
        start_frame = end_frame - len(track) + 1
        if start_frame < 0:
            raise ValueError('bad start frame: {}'.format(start_frame))
        arr[i, start_frame:end_frame+1] = track
        arr_mask[i, start_frame:end_frame+1] = True

    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1], frame_on=False)
    ax.axis('off')
    ax.imshow(video[0], cmap='gray', interpolation='nearest', vmin=0, vmax=255, animated=True, aspect='equal')
    ax.autoscale(False)
    if title:
        ax.set_title(title)

    all_artists = []
    print('{} frames'.format(len(video)))
    for frame_idx in range(len(video)):
        frame_artists = []

        # plot video frame
        im = ax.imshow(video[frame_idx], cmap='gray', interpolation='nearest',
                       vmin=0, vmax=255, animated=True, aspect='equal')
        frame_artists.append(im)

        # get trajectories that have a point in the current frame
        track_indices = np.flatnonzero(arr_mask[:, frame_idx])
        # plot trajectories
        for track_idx in track_indices:
            # get chunk of track that should be plotted
            track = arr[track_idx, :frame_idx+1]
            mask = arr_mask[track_idx, :frame_idx+1]
            track = track[mask]

            # only plot chunks of tracks where there is more than one (x,y) coordinate
            if len(track) > 1:
                xs, ys = track.T
                # plot head of trajectory
                scat = ax.scatter(xs[-1], ys[-1], s=40, c='b', marker='+')
                # plot lines connecting points
                lines = ax.plot(xs, ys, '-', c='r', lw=1.0)
                # add scatter point and lines to artists in frame
                frame_artists.append(scat)
                frame_artists.extend(lines)

        # show frame number
        text = ax.text(0.01, 0.01, 'Frame {}/{}'.format(frame_idx+1, len(video)), transform=ax.transAxes)
        frame_artists.append(text)

        all_artists.append(frame_artists)

    ani = animation.ArtistAnimation(fig, all_artists, repeat=True)

    writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)  #, extra_args=['-vcodec', 'libx24'])
    ani.save(fname, writer=writer,  savefig_kwargs={'facecolor':'black'}, dpi=dpi)
    plt.close(fig)


