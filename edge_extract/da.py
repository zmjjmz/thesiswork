from __future__ import division
import cv2
import numpy as np
import random


def build_transformation_matrix(
        imsize, theta, offset, flip, scale, shear, stretch):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x0, y0 = np.array(imsize) / 2

    # apply all transforms around the center of the image
    center_transform = np.array([[1, 0, x0],
                                 [0, 1, y0],
                                 [0, 0,  1]], dtype=np.float32)

    stretch_transform = np.array(
        [[stretch[0], 0,                  0],
         [0,          stretch[1],         0],
         [0,          0,                  1]], dtype=np.float32)

    scale_transform = np.array(
        [[scale, 0,     0],
         [0,     scale, 0],
         [0,     0,     1]], dtype=np.float32)

    shear_transform = np.array(
        [[1,        shear[0], 0],
         [shear[1], 1,        0],
         [0,        0,        1]], dtype=np.float32)

    rotate_transform = np.array(
        [[cos_theta,  sin_theta, 0],
         [-sin_theta, cos_theta, 0],
         [0,          0,         1]], dtype=np.float32)

    # reflect about x and y
    if flip[0] and flip[1]:
        flip_transform = np.array(
            [[-1,  0, 0],
             [ 0, -1, 0],
             [ 0,  0, 1]], dtype=np.float32)
    # reflect about only x
    elif flip[0] and not flip[1]:
        flip_transform = np.array(
            [[ 1,  0, 0],
             [ 0, -1, 0],
             [ 0,  0, 1]], dtype=np.float32)
    # reflect about only y
    elif not flip[0] and flip[1]:
        flip_transform = np.array(
            [[-1,  0, 0],
             [ 0,  1, 0],
             [ 0,  0, 1]], dtype=np.float32)
    # do not reflect
    else:
        flip_transform = np.eye(3, dtype=np.float32)

    # undo the initial centering
    uncenter_transform = np.array(
        [[1, 0, -x0],
         [0, 1, -y0],
         [0, 0,  1]], dtype=np.float32)

    # need to do translation last or it will amplify the others
    translate_transform = np.array(
        [[1, 0, offset[0]],
         [0, 1, offset[1]],
         [0, 0, 1       ]], dtype=np.float32)

    # apply the transformations in the correct order
    M = center_transform.dot(
        stretch_transform).dot(
        scale_transform).dot(
        shear_transform).dot(
        rotate_transform).dot(
        flip_transform).dot(
        uncenter_transform).dot(
        translate_transform)[:2, :]  # remove the bottom row

    return M


def generate_transformations(
        samples, imsize, allow_rotation, rotation_min, rotation_max,
        max_offset, min_offset, max_scale, min_scale, max_shear, min_shear,
        max_stretch, min_stretch, flip_horizontal, flip_vertical):
    # E [0, 360), uniform
    radians = np.random.randint(rotation_min, high=rotation_max, size=samples) * (
        np.pi / 180) if allow_rotation else np.zeros(samples)

    # E [min, max], uniform
    offsets = (max_offset - min_offset) * np.random.random(
        size=(2 * samples)).reshape(samples, 2) - max_offset

    # E [min, max], log-uniform
    scales = np.e ** (np.log(max_scale) + (np.log(min_scale) - np.log(
        max_scale)) * np.random.random(size=samples))

    # E {True, False}, bernoulli
    flips_h = np.random.choice(
        [True, False], size=samples, replace=True,
        p=(flip_horizontal, 1 - flip_horizontal))
    flips_v = np.random.choice(
        [True, False], size=samples, replace=True,
        p=(flip_vertical, 1 - flip_vertical))
    flips = [(flip_x, flip_y) for flip_x, flip_y in zip(flips_h, flips_v)]

    # E [min, max], uniform
    shears = np.tan(np.deg2rad((max_shear - min_shear) * np.random.random(
        size=(2 * samples)).reshape(samples, 2) - max_shear))

    # E [min, max], log-uniform
    stretches = np.e ** (np.log(max_stretch) + (np.log(min_stretch) - np.log(
        max_stretch)) * np.random.random(
            size=(2 * samples)).reshape(samples, 2))

    assert len(radians) == len(offsets) == len(scales) == len(flips) == len(
        shears) == len(stretches), 'need equal length for all parameters'
    transformations = [
        build_transformation_matrix(
            imsize, rotation, offset, flip, scale, shear, stretch)
        for rotation, offset, flip, scale, shear, stretch in zip(
            radians, offsets, flips, scales, shears, stretches)]

    return transformations

def generate_pixel_transformations(n_transforms,
        intensity_invert, brightness_multiply_prob, brightness_multiply_range, # color
        blur_prob, kernel_sizes, sigma_range): # gaussian blur (always symmetric)
    # sample a transformation
    transform_specs = []
    for i in range(n_transforms):
        transform_spec = {}

        # COLOR
        transform_spec['color'] = {}
        transform_spec['color']['intensity_invert'] = True if np.random.rand() < intensity_invert else False

        if np.random.rand() < brightness_multiply_prob:
            transform_spec['color']['brightness_multiply'] = np.random.uniform(*brightness_multiply_range)

        # FILTERING
        transform_spec['filter'] = {}
        ksize = random.choice(kernel_sizes)
        transform_spec['filter']['do_blur'] = np.random.rand() < blur_prob
        transform_spec['filter']['ksize'] = (ksize, ksize)
        transform_spec['filter']['sigmaX'] = np.random.uniform(*sigma_range)

        transform_specs.append(transform_spec)
    return transform_specs

def apply_transformation_matrix(X, M, border_mode):
    dsize = X.shape[::-1]
    return cv2.warpAffine(X, M, dsize, 0, 0, border_mode)

def apply_filter_transform(X, transform_spec):
    # just filter transform for now
    if transform_spec['do_blur']:
        return cv2.GaussianBlur(X, ksize=transform_spec['ksize'], sigmaX=transform_spec['sigmaX'])
    else:
        return X

def apply_color_transform(X, transform_spec):
    # assume there's no color channels, i.e. len(X.shape) == 2
    # color inversion
    if transform_spec['intensity_invert']:
        # take the maximum value, and apply lambda x: max_val - x
        X = np.max(X) - X

    # adjust brightness
    if 'brightness_multiply' in transform_spec:
        X *= transform_spec['brightness_multiply']
    return X


def transform_batch(Xb, yb, params, border_mode, transform_y=False, y_coords=True):
    transformations = generate_transformations(
        Xb.shape[0], Xb[0, 0].shape, **params['affine'])
    color_trans = generate_pixel_transformations(
            Xb.shape[0], **params['pixel'])
    Xbc = np.copy(Xb)
    if transform_y:
        ybc = np.copy(yb)
    for i, (M, ts) in enumerate(zip(transformations,color_trans)):
        if Xbc[i].shape[0] == 1:
            if params['doaffine']:
                Xbc[i, 0] = apply_transformation_matrix(Xbc[i, 0], M, border_mode)
            if params['dopixel']:
                Xbc[i, 0] = apply_color_transform(Xbc[i, 0], ts['color'])
                Xbc[i, 0] = apply_filter_transform(Xbc[i, 0], ts['filter'])
        elif Xbc[i].shape[0] == 3:
            Xbc[i, 0] = apply_transformation_matrix(Xbc[i, 0], M, border_mode)
            Xbc[i, 1] = apply_transformation_matrix(Xbc[i, 1], M, border_mode)
            Xbc[i, 2] = apply_transformation_matrix(Xbc[i, 2], M, border_mode)
        else:
            raise NotImplementedError
        if transform_y and params['doaffine']:
            # assume we've been given an ndarray of x,y coords
            # first axis is batch axis, second is the coords, third is x, y
            # if the transformation is (3,2), then we need to hstack [0,0,1] and
            # stack a ones vector to the points given
            if y_coords:
                coords_M = np.vstack([M,np.array([0,0,1],dtype=np.float32).reshape(1,-1)])
                yb_homog = np.hstack([ybc[i],np.ones((ybc[i].shape[0],1))])
                transformed = np.dot(coords_M,yb_homog.T).T[:,:-1] # take off the homog part
                ybc[i] = transformed
            else:
                # assume y is instead a segmentation image or something
                # for each channel, apply the transformation matri
                for c in range(ybc.shape[1]):
                    ybc[i, c] = apply_transformation_matrix(ybc[i, c], M, border_mode)

    if transform_y:
        return Xbc, ybc
    else:
        return Xbc, yb


def transform(X, y, transform_y=False, y_coords=True, border_mode=cv2.BORDER_REPLICATE):
    params = {}
    params['doaffine'] = False
    params['dopixel'] = True
    params['affine'] = {
        'allow_rotation': False,
        'rotation_min':0,
        'rotation_max':0,
        'min_offset': 0,
        'max_offset':0 ,
        'min_scale': 1,
        'max_scale': 1,
        #'min_scale': 1 / 1.6,
        #'max_scale':     1.6,
        'min_shear':  0,
        'max_shear':  0,
        #'min_shear':  -20,
        #'max_shear':   20,
        'min_stretch': 1,
        'max_stretch': 1,
        #'min_stretch': 1 / 1.3,
        #'max_stretch':     1.3,
        'flip_horizontal': 0,
        'flip_vertical': 0,
    }
    params['pixel'] = {
        'intensity_invert':0.3,
        'brightness_multiply_prob':0.4,
        'brightness_multiply_range':(0.7, 1.3),
        'blur_prob':0.3,
        'kernel_sizes':(3,5,7),
        'sigma_range':(0.7,1.5), # not arbitrary
    }

    return transform_batch(X, y, params, border_mode, transform_y=transform_y, y_coords=y_coords)
