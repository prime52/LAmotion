import math
import cv2
import numpy as np


def get_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dX, dY = x2 - x1, y2 - y1
    degree = math.degrees(math.atan2(-dY, dX))

    # Due to flip in degree. Will make up for real flip via rot_flag.
    if degree > 0.0:
        degree = 180.0 - degree
    return abs(degree)


def affine_mat(image, angle):
    image_size = (image.shape[1], image.shape[0])  # NumPy stores image matricies backwards
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    image_w2, image_h2 = image_size[0] * 0.5, image_size[1] * 0.5

    # Rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound, left_bound = max(x_pos), min(x_neg)
    top_bound, bot_bound = max(y_pos), min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # Translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Matrix for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    return affine_mat, new_w, new_h


def rotate_image(image, angle, point1=None, point2=None):
    mat, new_w, new_h = affine_mat(image, angle)

    # Apply the transform
    result = cv2.warpAffine(
        image,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    if point1 is None:
        output = result
    else:
        # Form of points from https://stackoverflow.com/questions/44378098/trouble-getting-cv-transform-to-work
        r_point1 = cv2.transform(np.array([[[point1[0], point1[1]]]]), mat)[0][0]
        r_point2 = cv2.transform(np.array([[[point2[0], point2[1]]]]), mat)[0][0]
        output = result, r_point1, r_point2

    return output
