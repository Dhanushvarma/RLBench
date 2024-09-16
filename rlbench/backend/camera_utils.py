import numpy as np

def pose_inv(pose):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose (np.array): 4x4 matrix for the pose to inverse

    Returns:
        np.array: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv

def project_points_from_world_to_camera(points, world_to_camera_transform, camera_height, camera_width):
    """
    Helper function to project a batch of points in the world frame
    into camera pixels using the world to camera transformation.

    Args:
        points (np.array): 3D points in world frame to project onto camera pixel locations. Should
            be shape [..., 3].
        world_to_camera_transform (np.array): 4x4 Tensor to go from robot coordinates to pixel
            coordinates.
        camera_height (int): height of the camera image
        camera_width (int): width of the camera image

    Return:
        pixels (np.array): projected pixel indices of shape [..., 2]
    """
    assert points.shape[-1] == 3  # last dimension must be 3D
    assert len(world_to_camera_transform.shape) == 2
    assert world_to_camera_transform.shape[0] == 4 and world_to_camera_transform.shape[1] == 4

    # convert points to homogenous coordinates -> (px, py, pz, 1)
    ones_pad = np.ones(points.shape[:-1] + (1,))
    points = np.concatenate((points, ones_pad), axis=-1)  # shape [..., 4]

    # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do robot frame to pixels transform
    mat_reshape = [1] * len(points.shape[:-1]) + [4, 4]
    cam_trans = world_to_camera_transform.reshape(mat_reshape)  # shape [..., 4, 4]
    pixels = np.matmul(cam_trans, points[..., None])[..., 0]  # shape [..., 4]

    # re-scaling from homogenous coordinates to recover pixel values
    # (x, y, z) -> (x / z, y / z)
    pixels = pixels / pixels[..., 2:3]
    pixels = pixels[..., :2].round().astype(int)  # shape [..., 2]

    # swap first and second coordinates to get pixel indices that correspond to (height, width)
    # and also clip pixels that are out of range of the camera image
    pixels = np.concatenate(
        (
            pixels[..., 1:2].clip(0, camera_height - 1),
            pixels[..., 0:1].clip(0, camera_width - 1),
        ),
        axis=-1,
    )

    return pixels


def transform_from_pixels_to_world(pixels, depth_map, camera_to_world_transform):
    """
    Helper function to take a batch of pixel locations and the corresponding depth image
    and transform these points from the camera frame to the world frame.

    Args:
        pixels (np.array): pixel coordinates of shape [..., 2]
        depth_map (np.array): depth images of shape [..., H, W, 1]
        camera_to_world_transform (np.array): 4x4 Tensor to go from pixel coordinates to world
            coordinates.

    Return:
        points (np.array): 3D points in robot frame of shape [..., 3]
    """

    # make sure leading dimensions are consistent
    pixels_leading_shape = pixels.shape[:-1]
    depth_map_leading_shape = depth_map.shape[:-3]
    assert depth_map_leading_shape == pixels_leading_shape

    # sample from the depth map using the pixel locations with bilinear sampling
    pixels = pixels.astype(float)
    im_h, im_w = depth_map.shape[-2:]
    depth_map_reshaped = depth_map.reshape(-1, im_h, im_w, 1)
    z = bilinear_interpolate(im=depth_map_reshaped, x=pixels[..., 1:2], y=pixels[..., 0:1])
    z = z.reshape(*depth_map_leading_shape, 1)  # shape [..., 1]

    # form 4D homogenous camera vector to transform - [x * z, y * z, z, 1]
    # (note that we need to swap the first 2 dimensions of pixels to go from pixel indices
    # to camera coordinates)
    cam_pts = [pixels[..., 1:2] * z, pixels[..., 0:1] * z, z, np.ones_like(z)]
    cam_pts = np.concatenate(cam_pts, axis=-1)  # shape [..., 4]

    # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do camera to robot frame transform
    mat_reshape = [1] * len(cam_pts.shape[:-1]) + [4, 4]
    cam_trans = camera_to_world_transform.reshape(mat_reshape)  # shape [..., 4, 4]
    points = np.matmul(cam_trans, cam_pts[..., None])[..., 0]  # shape [..., 4]
    return points[..., :3]

def bilinear_interpolate(im, x, y):
    """
    Bilinear sampling for pixel coordinates x and y from source image im.
    Taken from https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def get_transform_matrix(extrinsics, intrinsics):
    R = extrinsics
    K = intrinsics
    K_exp = np.eye(4)
    K_exp[:3, :3] = K

    # Takes a point in world, transforms to camera frame, and then projects onto image plane.
    transform = K_exp @ pose_inv(R)

    return transform

