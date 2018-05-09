import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def show_matched_points(img1, img2, points1=None, points2=None, color_channels='green-magenta'):
    """
    Show corresponding points connected by a line.

    Parameters
    ----------
    img1 : array, shape (H, W)
        Input image one.
    img2 : array, shape (H, W)
        Input image two.
    points1 : array, shape (M, 2)
        Coordinates of points in image one, specified as an M-by-2 matrix of M number of [x y] coordinates.
    points2 : array, shape (M, 2)
        Coordinates of points in image two, specified as an M-by-2 matrix of M number of [x y] coordinates.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(tight_layout=True)

    # Plot difference between images
    if color_channels == 'green-magenta':
        composite = np.dstack((img2, img1, img2))
    elif color_channels == 'red-cyan':
        composite = np.dstack((img1, img2, img2))
    else:
        raise ValueError()

    ax.grid(False)
    ax.imshow(composite, interpolation='nearest')
    ax.set_autoscale_on(False)

    if points1 is not None and points2 is not None:
        # Plot first points
        ax.scatter(points1[:, 0], points1[:, 1], s=20, marker='o', facecolors='none', edgecolors='r', label='Points 1')
        # Plot second points
        ax.scatter(points2[:, 0], points2[:, 1], s=30, c='g', marker='+', linewidths=1, label='Points 2')
        # Plot line between points
        xs = np.array([points1[:, 0], points2[:, 0]])
        ys = np.array([points1[:, 1], points2[:, 1]])
        ax.plot(xs, ys, color='yellow', linestyle='solid', linewidth=1)
        ax.legend(loc='best', frameon=True)

    plt.show()
    plt.close(fig)


def show_transforms(transforms):
    """Show the parameters of multiple transforms in the temporal domain.

    Each parameter of the transform is plotted over time, with the following notation:
    =========================
    || a | b | translate_x ||
    ||=====================||
    || c | d | translate_y ||
    =========================

    Parameters
    ----------
    transforms : array, shape (n_transforms, 2, 3)
        Transforms.
    """
    a = transforms[:, 0, 0]
    b = transforms[:, 0, 1]
    c = transforms[:, 1, 0]
    d = transforms[:, 1, 1]
    translate_x = transforms[:, 0, 2]
    translate_y = transforms[:, 1, 2]

    fig, axes = plt.subplots(nrows=3, ncols=2, tight_layout=True)
    axes[0, 0].plot(a)
    axes[0, 1].plot(b)
    axes[1, 0].plot(c)
    axes[1, 1].plot(d)
    axes[2, 0].plot(translate_x)
    axes[2, 1].plot(translate_y)

    for ax in axes.flat:
        ax.set_xlim(0, len(transforms))

    axes[0, 0].set_title('a')
    axes[0, 1].set_title('b')
    axes[1, 0].set_title('c')
    axes[1, 1].set_title('d')
    axes[2, 0].set_title('translate_x')
    axes[2, 1].set_title('translate_y')

    plt.show()
    plt.close(fig)


################
# Animations

def show_videos(video1, video2, title1="", title2="", fps=10):
    """Show two videos side by side.

    Parameters
    ----------
    video1 : array, shape (n_frames, height, width)
        First video.
    video2 : array, shape (n_frames, height, width)
        Second video.
    title1 : str
        Title of first video.
    title2 : str
        Title of second video.
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, tight_layout=True)
    for _ax, title in zip(ax, [title1, title2]):
        _ax.set_title(title)
        _ax.grid(False)

    ax[0].imshow(video1[0], cmap='gray', aspect='equal')
    ax[1].imshow(video1[1], cmap='gray', aspect='equal')

    ims = []
    for frame1, frame2 in zip(video1, video2):
        im1 = ax[0].imshow(frame1, cmap='gray', animated=True, aspect='equal')
        im2 = ax[1].imshow(frame2, cmap='gray', animated=True, aspect='equal')
        ims.append([im1, im2])

    ani = animation.ArtistAnimation(fig, ims, blit=False, interval=(1/30)*100)
    plt.show()
    plt.close(fig)
