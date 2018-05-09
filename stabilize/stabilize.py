import multiprocessing
import os

import cv2
import numpy as np

import cilia.utils.video as video_utils
from old_code import features
from . import plot


def stabilize(frames, mask=None, crop=False):
    """
    Stabilize a sequence of images using point matching with dense optical flow.

    Parameters
    -----------
    frames : array, shape (num_frames, height, width)
        Sequence of images to stabilize.
    mask : array of bool with shape (height, width), optional
        If specified, ignore pixels with value True when matching points. The dimensions
        must match the last two dimensions of `frames`.
    crop : bool, optional
        If crop is True, crop the stabilized frames so that no border is shown. If crop is False, return
        the stabilized frames with the same dimensions as the original frames.

    Returns
    -------
    stabilized_frames : array
        Sequence of stabilized images. If `crop` is False, will have the same dimensions as `frames`.

    See also
    -----------
        http://www.mathworks.com/help/vision/examples/video-stabilization-using-point-feature-matching.html
    """
    num_frames, height, width = frames.shape

    # Compute optical flow between all frames. Optical flow will be a
    # (num_frames-1, 2, height, width) array
    with multiprocessing.Pool() as pool:
        optflow = np.array(pool.starmap(features.optical_flow_fb, zip(frames, frames[1:])))
        optflow = np.rollaxis(optflow, -1, 1)
    # flow = features.optical_flow(frames)

    # Estimate the rigid transformation between all frames.
    # flow[0] contains the x components of optical flow and flow[1]
    # contains the y components of optical flow. We reverse the np.mgrid
    # call so that ptsA[0] has x components and ptsA[1] has y components.
    transforms = np.zeros(shape=(len(optflow), 2, 3), dtype=np.float64)
    for idx, flow in enumerate(optflow):
        ptsA = np.mgrid[0:height, 0:width][::-1].astype(np.float64)
        ptsB = ptsA + flow

        # Convert the (2, height, width) point sets to (num_pixels, 2) arrays of (x,y) pixels.
        ptsA = grid_to_coords(ptsA[0], ptsA[1])
        ptsB = grid_to_coords(ptsB[0], ptsB[1])
        transforms[idx] = cv2.estimateRigidTransform(ptsB, ptsA, fullAffine=False)

        print('Plotting')
        plot.show_matched_points(frames[idx], frames[idx+1], ptsA[::100], ptsB[::100])
        plot.show_matched_points(frames[idx], cv2.warpAffine(frames[idx+1], transforms[idx], dsize=(width, height)))


    # Create the cumulative product of the transformation matrices. The transformation matrix
    # cum_transforms[i] is the transform to align frame i+1 with frame i.
    transforms = to_homogeneous(transforms)
    cum_transforms = np.zeros_like(transforms)
    cum_transforms[0] = transforms[0]
    for i in range(1, len(transforms)):
        cum_transforms[i] = np.dot(transforms[i], cum_transforms[i-1])
    # plot.show_transforms(cum_transforms)

    # Warp each image with its cumulative transform
    cum_transforms = cum_transforms[:, :2, :]
    corrected_frames = np.zeros_like(frames)
    corrected_frames[0] = frames[0]
    for i in range(1, len(frames)):
        corrected_frames[i] = cv2.warpAffine(frames[i], cum_transforms[i-1], dsize=(width, height))

    plot.show_videos(frames, corrected_frames, "Original", "Stabilized")

    if crop:
        corrected_frames = __crop_stabilized_frames(corrected_frames)

    return corrected_frames


def to_homogeneous(transform):
    """
    Convert (2, 3) transform(s) to (3, 3) homogeneous transform(s) by concatenating
    [0 0 1] to the bottom of the matrix.

    Parameters
    ----------
    transform : array, shape (2, 3) or (n_transforms, 2, 3)
        Transform or multiple transforms.

    Returns
    -------
    homog_transforms : array, shape (3, 3) or (n_transforms, 3, 3)
        Homogeneous transform(s).
    """
    return np.append(transform, [[[0,0,1]]]*len(transform), axis=-2)


def grid_to_coords(grid_x, grid_y):
    """Transform grids of x coordinates and y coordinates to their (x, y) pairs.

    Parameters
    ----------
    grid_x : array, shape (height, width)
        2-d grid of x components.
    grid_y : array, shape (height, width)
        2-d grid of y components.

    Returns
    -------
    coords : array, shape (height*width, 2)
        (x,y) points.
    """
    return np.array((grid_x.flatten(), grid_y.flatten())).T


def __crop_stabilized_frames(frames):
    """
    Crop sequence of frames to remove borders. Any pixel having value 0 is considered a border.

    Arguments
    -----------
        frames (array: txhxw): Frames to be cropped.

    Returns
    -----------
        cropped_frames (array: txhxw): Cropped frames.
    """
    # create binary image with borders of all frames
    aggregate_borders = ~np.any(~frames.astype(np.bool), axis=0)

    # extract polygon of shape
    _, contours, _ = cv2.findContours(
        aggregate_borders.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    vertices = np.squeeze(contours[0])

    # get corners of polygons closest to corners of original image
    height, width = frames.shape[1:]
    poly_top_left = vertices[np.argmin([np.linalg.norm(np.array([0,0])-vert) for vert in vertices])]
    poly_top_right = vertices[np.argmin([np.linalg.norm(np.array([width,0])-vert) for vert in vertices])]
    poly_bottom_left = vertices[np.argmin([np.linalg.norm(np.array([0, height])-vert) for vert in vertices])]
    poly_bottom_right = vertices[np.argmin([np.linalg.norm(np.array([width, height])-vert) for vert in vertices])]

    # compute rectangle coordinates for new frame size
    y_min, y_max = max(poly_top_left[1], poly_top_right[1]), min(poly_bottom_left[1], poly_bottom_right[1])
    x_min, x_max = max(poly_top_left[0], poly_bottom_left[0]), min(poly_top_right[0], poly_bottom_right[0])

    # crop original frames to new size
    cropped_frames = frames[:, y_min:y_max, x_min:x_max]

    return cropped_frames


#################################################################################
# Functions used in pipeline
def stabilize_videos(directory):
    """Stabilize all .avi files in the directory.

    Create a stabilized version of the video in the same directory as original.
    """
    avis = [os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith('.avi') and not f.endswith('_stabilized.avi')]
    for avi_file in avis:
        print(avi_file)
        new_avi_file = '{}_stabilized{}'.format(*os.path.splitext(avi_file))

        # Read video and stabilize.
        frames, fps = video_utils.read(avi_file, return_fps=True)
        frames = frames[:500:5]
        # mask = roi_utils.polys2mask(frames, video['rois'])
        stabilized_frames = stabilize(frames)

        # Save stabilized video to same directory as original video.
        print('Saving "{}"'.format(new_avi_file))
        video_utils.write(stabilized_frames, new_avi_file, fps, overwrite=True)
