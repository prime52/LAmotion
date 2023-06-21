# coding=utf-8
'''

(input)
dicom cine data

(output)
pkl file saving
LA1, 2, 3, ..., N
- estimated long axis image
- long axis image

html file saving

author: Yoon-Chul Kim, Minwoo Kim
'''

import pathlib
import re
import sys, os
import glob
import math
import webbrowser

import cv2 as cv
import matplotlib as mpl
mpl.use('Qt5Agg')
import numpy as np
import plotly
import pydicom as dicom
import pickle
from skimage.transform import rotate
from skimage import exposure
from scipy.interpolate import interpn
import sympy
from sympy import Point, Line, Segment, Plane, Point3D, Ray3D
from sympy.geometry import Line3D, Segment3D

import FileSlider
import Util

from RotateAndCrop import get_angle, rotate_image

plotly.offline.init_notebook_mode(connected=True)


''' input data '''
DATA_PATH = "dcm/KAG_2ndAnnual_validate/validate_batch0"

FLAG_PROCESS_ALL_PATIENTS = True  # if True, opens up entire pkl files in the folder!!
patient_to_process = "DET0000301"

''' output data '''
# save_pkl_dir = "data/pkl_la5"
save_pkl_dir = "data/KAG_2ndAnnual/pkl_la_validate"

save_pkl = True
FLAG_PROCESS_ALL_PHASES = False  # if False, process only phase 0
FLAG_PRINT_SPECIFICS = True

# https://bastibe.de/2013-05-30-speeding-up-matplotlib.html
# https://matplotlib.org/3.1.1/users/event_handling.html
# https://matplotlib.org/3.1.1/gallery/widgets/slider_demo.html

# https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom.html
# http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html
# http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_10.7.html#sect_10.7.1.3
# https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032

# https://nipy.org/nibabel/dicom/dicom_orientation.html


def get_trans_mat3D(dcm: dicom.dataset.FileDataset) -> np.ndarray:
    """
    with Image Position
    :param dcm:
    :return: 4 x 4 array (in homogeneous form)
    """
    position = Util.get_pos(dcm)

    pixel_spacing = dcm.PixelSpacing
    c_res = pixel_spacing[1]
    r_res = pixel_spacing[0]
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    row_cos_vec, col_cos_vec = orientation[:3], orientation[3:]
    trans_mat = np.array([[row_cos_vec[0] * c_res, col_cos_vec[0] * r_res, 0.0, position[0]],
                          [row_cos_vec[1] * c_res, col_cos_vec[1] * r_res, 0.0, position[1]],
                          [row_cos_vec[2] * c_res, col_cos_vec[2] * r_res, 0.0, position[2]],
                          [0.0, 0.0, 0.0, 1.0]])
    return trans_mat


def get_trans_mat2D_fixed(dcm: dicom.dataset.FileDataset) -> np.ndarray:
    """
    creates a matrix that projects to XY plane without using Image Position data
    :param dcm:
    :return: 2 x 2 array
    """
    pixel_spacing = dcm.PixelSpacing
    # print("pixel_spacing_z", pixel_spacing)
    # pixel_spacing_z [1.37291, 1.37291]
    c_res = pixel_spacing[1]
    r_res = pixel_spacing[0]

    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    # print("orientation_z", orientation)
    # orientation_z (-0.68662, 0.727017, 0.0, 0.0, -0.0, -1.0)

    row_cos_vec = orientation[:3]
    col_cos_vec = orientation[3:]

    # row_cos_vec (-0.68662, 0.727017, 0.0)
    # col_cos_vec (0.0, -0.0, -1.0)

    position = Util.get_pos(dcm)

    trans_mat2D = np.array([[row_cos_vec[0] * c_res, col_cos_vec[0] * r_res, position[0]],
                            [row_cos_vec[1] * c_res, col_cos_vec[1] * r_res, position[1]],
                            [row_cos_vec[2] * c_res, col_cos_vec[2] * r_res, position[2]]])
    return trans_mat2D


def thru_plane_position(dcm: dicom.dataset.FileDataset) -> np.ndarray:
    """Gets spatial coordinate of image origin whose axis
    is perpendicular to image plane.
    """
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    position = tuple((float(p) for p in dcm.ImagePositionPatient))

    row_vec, col_vec = orientation[:3], orientation[3:]
    # get x vec, y vec of orientation plane
    # e.g. orientation = [1, 2, 3, 4, 5, 6]
    # row_vec = [1 ,2 ,3], col_vec = [4, 5, 6]

    normal_vector = np.cross(row_vec, col_vec)
    # get cross product of the x vec & y vec
    # = a vector perpendicular to both x & y
    # normal_vector = [-3, 6, -3]

    slice_pos = np.dot(position, normal_vector)
    return slice_pos


def get_spacing_between_slices(dcm_files: dicom.dataset.FileDataset) -> np.ndarray:
    """
    Get difference of each slice_pos' and calculate mean.
    It's usually 10, with some outliers.
    :param dcm_files:
    :return:
    """
    spacings = np.diff([thru_plane_position(dcm) for dcm in dcm_files])
    spacing_between_slices = np.mean(spacings)
    return spacing_between_slices


def get_sorted_SA_plane_names(file_names_list: list) -> list:
    """
    Sorting list of dcm files
    by key=lambda x: thru_plane_position(x[0]))
    :param file_names_list: dcm files list
    :return: list
    """
    dcm_files = []
    for fname in file_names_list:
        dfile = dicom.read_file(str(fname))
        dcm_files.append((dfile, fname))
    dcm_files = sorted(dcm_files, key=lambda x: thru_plane_position(x[0]))
    # print(dcm_files)
    _, sorted_SA_file_names = zip(*dcm_files)

    return sorted_SA_file_names


def get_plotly_planes_list(file_names_list, n_planes: int = sys.maxsize) -> list:
    dcm_files = []
    planes_list = []
    plane3ds_list = []
    cine_img_arr = []
    n_slice = min(len(file_names_list), n_planes)

    for fname in file_names_list:
        dfile = dicom.read_file(str(fname))
        dcm_files.append(dfile)

    for dfile in dcm_files:
        cine_img_arr.append(dfile.pixel_array.astype(np.float32))

    n_row, n_col = cine_img_arr[0].shape
    cine_img_stack = np.dstack(cine_img_arr)
    # SA image stack => 3D

    for i in range(n_slice):
        trans_mat = get_trans_mat3D(dcm_files[i])
        # a translation mat that enables following:
        # a point in 2D image plane (k, j) => a point in 3D space (Px, Py, Pz)
        points_in_3d_coord = np.array([[trans_mat @ np.array([k, j, 0.0, 1.0]) for k in range(n_col)] for j in range(n_row)])

        point1_in_3d_coord = np.array(trans_mat @ np.array([0.0, 0.0, 0.0, 1.0]))
        point1_in_point3d = Point3D(point1_in_3d_coord[0], point1_in_3d_coord[1], point1_in_3d_coord[2])

        point2_in_3d_coord = np.array(trans_mat @ np.array([1.0, 0.0, 0.0, 1.0]))
        point2_in_point3d = Point3D(point2_in_3d_coord[0], point2_in_3d_coord[1], point2_in_3d_coord[2])

        point3_in_3d_coord = np.array(trans_mat @ np.array([0.0, 1.0, 0.0, 1.0]))
        point3_in_point3d = Point3D(point3_in_3d_coord[0], point3_in_3d_coord[1], point3_in_3d_coord[2])

        plane_in_3d_coord = Plane(point1_in_point3d, point2_in_point3d, point3_in_point3d)

        plane3ds_list.append(plane_in_3d_coord)

        # @ : https://www.python.org/dev/peps/pep-0465/
        # automatically transpose (1,4) into (4,1) when needed for calculation
        # and also transpose the calculation answer into (1,4)

        # k : Column index to the image plane
        # j : Row index to the image plane
        # points_in_3d_coord : The coordinates of the voxel (i,j) in the frame's image plane
        #           in units of mm
        # points_in_3d_coord = n_row x n_col x (4 x 1) = n_row x n_col x [Px, Py, Pz, 1]

        # a point in 3D space (Px, Py, Pz) => a surface
        plane = Util.get_plane(points_in_3d_coord[:, :, 0], points_in_3d_coord[:, :, 1], points_in_3d_coord[:, :, 2], cine_img_stack[:, :, i])
        planes_list.append(plane)

    return planes_list


def get_file_names_lists(dataset_name: str, patient_number: str, phase_number: int) -> tuple:

    if dataset_name == 'CAT':
        # Usage example:
        # N_PATIENT, N_PHASE = "dcm\\DET0003001", 0
        la_file_names = "*_LA*_ph" + str(phase_number) + ".dcm"
        sa_file_names = "*_SA*_ph" + str(phase_number) + ".dcm"
        la_dir_path = pathlib.Path(patient_number)
        sa_dir_path = pathlib.Path(patient_number)

        print( f'patient_number : { patient_number}, LA images dir : {la_dir_path}' )

        la_file_names_list = la_dir_path.glob(la_file_names)
        sa_file_names_list = sa_dir_path.glob(sa_file_names)

    elif dataset_name == 'KAG':

        path1 = os.path.join(patient_number, 'study')

        acq_list = glob.glob(path1+'/*')

        print(acq_list)

        dcm_LA_list = []
        dcm_SA_list = []
        for jj, acq_dir in enumerate(acq_list):

            folder, acq_name = os.path.split(acq_dir)

            if "ch" in acq_name:
                # long axis data
                dcm_LA_list.append(glob.glob(acq_dir+'/*0001.dcm')[0])
            elif "sax" in acq_name:
                dcm_SA_list.append(glob.glob(acq_dir+'/*0001.dcm')[0])

        la_file_names_list = dcm_LA_list
        sa_file_names_list = dcm_SA_list

        # la_dir_path = pathlib.Path(path1)
        #
        # # Usage example:
        # # N_PATIENT, N_PHASE = "dcm\\DET0003001", 0
        # la_file_names = "*_LA*_ph" + str(phase_number) + ".dcm"
        # sa_file_names = "*_SA*_ph" + str(phase_number) + ".dcm"
        # la_dir_path = pathlib.Path(patient_number)
        # sa_dir_path = pathlib.Path(patient_number)
        #
        # print('LA images dir : %s' % la_dir_path)
        #
        # la_file_names_list = la_dir_path.glob(la_file_names)
        # sa_file_names_list = sa_dir_path.glob(sa_file_names)



    return la_file_names_list, sa_file_names_list


def get_est_la_plane_from_img_stack(points: list, interpolated_img_stack: np.ndarray) -> np.ndarray:
    assert (points[0] != points[1])

    def get_t_val_from_x(x_: np.ndarray) -> np.ndarray:
        return (x_ - points[0][1]) / (points[1][1] - points[0][1])

    def get_t_val_from_y(y_: np.ndarray) -> np.ndarray:
        return (y_ - points[0][0]) / (points[1][0] - points[0][0])

    def get_x_val(t_: np.ndarray) -> np.ndarray:
        return (1 - t_) * points[0][1] + t_ * points[1][1]

    def get_y_val(t_: np.ndarray) -> np.ndarray:
        return (1 - t_) * points[0][0] + t_ * points[1][0]

    epsilon = 1.0
    n_x, n_y, n_z = interpolated_img_stack.shape
    x = np.linspace(0, n_x - 1, n_x)
    y = np.linspace(0, n_y - 1, n_y)
    z = np.linspace(0, n_z - 1, n_z)

    if abs(points[1][1] - points[0][1]) < epsilon:
        t_range_for_x = [-sys.float_info.max, sys.float_info.max]
    else:
        t_range_for_x = [min(get_t_val_from_x(0), get_t_val_from_x(n_x - 1)),
                         max(get_t_val_from_x(0), get_t_val_from_x(n_x - 1))]

    if abs(points[1][0] - points[0][0]) < epsilon:
        t_range_for_y = [-sys.float_info.max, sys.float_info.max]
    else:
        t_range_for_y = [min(get_t_val_from_y(0), get_t_val_from_y(n_y - 1)),
                         max(get_t_val_from_y(0), get_t_val_from_y(n_y - 1))]

    t_range = [max(t_range_for_x[0], t_range_for_y[0]),
               min(t_range_for_x[1], t_range_for_y[1])]

    # By Pythagorean theorem
    n_t = int(np.sqrt(np.square(get_x_val(t_range[0]) - get_x_val(t_range[1])) +
                      np.square(get_y_val(t_range[0]) - get_y_val(t_range[1]))))

    if get_x_val(t_range[0]) > get_x_val(t_range[1]):
        t = np.linspace(t_range[1], t_range[0], n_t)
    else:
        print(f't_range: from {t_range[0]} to {t_range[1]}')
        t = np.linspace(t_range[0], t_range[1], n_t)

    new_z = np.linspace(0, n_z - 1, n_z)
    t, new_z = np.meshgrid(t, new_z)
    new_x, new_y = get_x_val(t), get_y_val(t)

    # print(f'new_x range = {new_x[0,0], new_x[-1,-1]}')
    # print(f'new_y range = {new_y[0, 0], new_y[-1, -1]}')
    # print(f'new_z range = {new_z[0, 0], new_z[-1, -1]}')
    # print(f'new_x shape = {new_x.shape}, new_y shape = {new_y.shape}, new_z shape = {new_z.shape}')

    la_plane = interpn((x, y, z), interpolated_img_stack, np.dstack((new_x, new_y, new_z)))

    return la_plane


def get_intersection_line3D(lhs_plane: plotly.graph_objs.Surface,
                            rhs_plane: plotly.graph_objs.Surface) -> sympy.Line3D:

    lhs_points = Point3D(lhs_plane.x[0, 0], lhs_plane.y[0, 0], lhs_plane.z[0, 0]), \
                 Point3D(lhs_plane.x[-1, -1], lhs_plane.y[-1, -1], lhs_plane.z[-1, -1]), \
                 Point3D(lhs_plane.x[0, -1], lhs_plane.y[0, -1], lhs_plane.z[0, -1])

    rhs_points = Point3D(rhs_plane.x[0, 0], rhs_plane.y[0, 0], rhs_plane.z[0, 0]), \
                 Point3D(rhs_plane.x[-1, -1], rhs_plane.y[-1, -1], rhs_plane.z[-1, -1]), \
                 Point3D(rhs_plane.x[0, -1], rhs_plane.y[0, -1], rhs_plane.z[0, -1])

    # https://plot.ly/python-api-reference/generated/plotly.graph_objects.Surface.html

    lhs_plane, rhs_plane = Plane(*lhs_points), Plane(*rhs_points)

    return lhs_plane.intersection(rhs_plane)


def get_intersection_points2D_fixed(intersection_points: list,
                                    trans_mat2D: np.ndarray) -> tuple:
    lhs_point, rhs_point = intersection_points

    # get x1 that trans_mat2D * x1 = new_lhs_point
    x1 = np.linalg.solve(trans_mat2D, np.array([float(lhs_point.x), float(lhs_point.y), float(lhs_point.z)]))
    x2 = np.linalg.solve(trans_mat2D, np.array([float(rhs_point.x), float(rhs_point.y), float(rhs_point.z)]))

    x, y = np.array([x1[0], x2[0]]), np.array([x1[1], x2[1]])
    return x, y


def get_intersection_points2D_with_img(intersection_points: list, plane_range: np.ndarray) -> tuple:
    x, y = intersection_points
    p1, p2 = Point(x[0], y[0]), Point(x[1], y[1])
    intersection_line = Line(p1, p2)

    points1 = Point(plane_range[0, 1], plane_range[0, 0]), Point(plane_range[0, 1], plane_range[1, 0])
    points2 = Point(plane_range[0, 1], plane_range[1, 0]), Point(plane_range[1, 1], plane_range[1, 0])
    points3 = Point(plane_range[1, 1], plane_range[1, 0]), Point(plane_range[1, 1], plane_range[0, 0])
    points4 = Point(plane_range[1, 1], plane_range[0, 0]), Point(plane_range[0, 1], plane_range[0, 0])

    line1, line2, line3, line4 = Segment(*points1), Segment(*points2), Segment(*points3), Segment(*points4)

    result = tuple(filter(lambda li: li != [], intersection_line.intersection(line1) + intersection_line.intersection(
        line2) + intersection_line.intersection(line3) + intersection_line.intersection(line4)))

    return (float(result[0].x), float(result[0].y)), (float(result[1].x), float(result[1].y))


def get_interpolated_img_stack(file_names_list: list) -> np.ndarray:
    dcm_files = []
    cine_img_arr = []
    n_slices = len(file_names_list)

    for fname in file_names_list:
        dfile = dicom.read_file(str(fname))
        dcm_files.append(dfile)

    for dfile in dcm_files:
        cine_img_arr.append(dfile.pixel_array.astype(np.float32))

    n_row, n_col = cine_img_arr[0].shape
    spacing_between_slices = get_spacing_between_slices(dcm_files)
    num_of_inserted_picture = int(round(spacing_between_slices / dcm_files[0].PixelSpacing[0]))

    cine_img_stack = np.dstack(cine_img_arr)
    n_extended_height = ((n_slices - 1) * num_of_inserted_picture + n_slices)

    interpolated_img_stack = []
    for i in range(n_row):
        resized_img = np.expand_dims(cv.resize(cine_img_stack[i], (n_extended_height, n_col),
                                               interpolation=cv.INTER_LINEAR), axis=0)
        interpolated_img_stack.append(resized_img)

    interpolated_img_stack = np.concatenate(interpolated_img_stack, axis=0)

    return interpolated_img_stack


def exists_or_mkdir(path, verbose=True):
    if not os.path.exists(path):
        if verbose:
            print("[*] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            print("[!] %s exists ..." % path)
        return True


def how_many_phases_in_folder(patient_folder_path):
    phase_num = 0
    look_for = patient_folder_path + '/*_LA1_ph*.dcm'
    for filename in glob.glob(look_for):
        phase_num += 1
    return phase_num


def main():
    # N_PATIENT, N_PHASE = "CAP_Challenge_Training_Set\\DET0000101", 30
    # N_PATIENT, N_PHASE = "dcm\\DET0000101", 30


    if FLAG_PROCESS_ALL_PATIENTS:
        patient_folders_list = os.listdir(DATA_PATH)
    else:
        patient_folders_list = [os.path.join(patient_to_process)]

    exists_or_mkdir(save_pkl_dir)

    for patient in patient_folders_list:
        # patient_folder_path = DATA_PATH + "/" + patient

        patient_folder_path = os.path.join(DATA_PATH, patient)
        print("patient_folder_path:", patient_folder_path)

        if FLAG_PROCESS_ALL_PHASES:
            number_of_phases = how_many_phases_in_folder(patient_folder_path)
        else:
            number_of_phases = 1
        # number_of_phases = how_many_phases_in_folder(patient_folder_path)

        print("Patient:", patient, "  Number of phases:", number_of_phases)

        if number_of_phases == 0:
            # NO LA IMAGES
            pass
        else:

            dataset = 'KAG'

            for phaseno in range(number_of_phases):
                print("Processing phase: [", phaseno, "/", number_of_phases-1, "]")
                print(f'patient_folder_path = {patient_folder_path}')

                la_file_names_list, sa_file_names_list = get_file_names_lists(dataset, patient_folder_path, phaseno)

                print(la_file_names_list, sa_file_names_list)

                la_file_names_list = sorted(la_file_names_list, key=Util.sort_by_plane_number)

                print(la_file_names_list)

                # sorting bunch of SA files by plane position (different key param than LA)
                sa_file_names_list = get_sorted_SA_plane_names(sa_file_names_list)

                print(la_file_names_list)
                print(sa_file_names_list)

                n_la_planes = len(la_file_names_list)
                n_sa_planes = len(sa_file_names_list)

                interpolated_img_stack = get_interpolated_img_stack(sa_file_names_list)

                # [file name, ... ]  => [plotly surface of that file name, ... ]
                sa_plotly_planes_list = get_plotly_planes_list(sa_file_names_list)

                la_img_list = []
                sa_img_list = [sa_plotly_planes_list[pln].surfacecolor for pln in range(n_sa_planes)]
                sa_intersection_points_list = []
                la_intersection_points_list = []
                estimated_la_img_list = []
                rot_flag = 0

                la_titles_list = [f"{patient_folder_path}_original_la{str(i + 1)}_ph{str(phaseno)}" for i in range(n_la_planes)]
                sa_num_list = [int((re.split(r'(\d+)', str(sa_file_names_list[i])))[-4]) for i in range(n_sa_planes)]
                sa_titles_list = [f"{patient_folder_path}_original_sa{sa_num}_ph{str(phaseno)}" for sa_num in sa_num_list]
                est_la_titles_list = [f"estimated_la{str(i + 1)}_ph{str(phaseno)}" for i in range(n_la_planes)]

                for i, la_file_name in enumerate(la_file_names_list):
                    print('LA image %d' % i)
                    la_plotly_planes_list = get_plotly_planes_list([la_file_name], 1)  # => [la_pln], 슬라이스 1개만 만들기.

                    la_dicom_file = Util.get_dicom_file(la_file_name)
                    la_trans_mat2D = get_trans_mat2D_fixed(la_dicom_file)  # ignore z & position data
                    la_plane_range = Util.get_plane_z_range(la_plotly_planes_list[0].surfacecolor)

                    estimated_la_img_list.append([])
                    sa_intersection_points_list.append([])
                    la_intersection_points_list.append([])

                    for j, sa_file_name in enumerate(sa_file_names_list):
                        sa_dicom_file = Util.get_dicom_file(sa_file_name)
                        sa_trans_mat2D = get_trans_mat2D_fixed(sa_dicom_file)
                        sa_plane_range = Util.get_plane_xy_range(sa_plotly_planes_list[j].surfacecolor)

                        # intersection of LA on SA
                        sa_intsct_line3D = get_intersection_line3D(sa_plotly_planes_list[j], la_plotly_planes_list[0])

                        # print(i, j, sa_intsct_line3D[0].points)

                        sa_intsct_points3D = sa_intsct_line3D[0].points  # Point3D(a1, b1, 0), Point3D(a2, b2, c2)

                        sa_intsct_points2D = get_intersection_points2D_fixed(sa_intsct_points3D, sa_trans_mat2D)
                        sa_intsct_points2D_with_img = get_intersection_points2D_with_img(sa_intsct_points2D, sa_plane_range)
                        sa_p1, sa_p2 = sa_intsct_points2D_with_img
                        sa_intersection_points_list[i].append([sa_p1, sa_p2])

                        # intersection of SA on LA
                        la_intsct_line3D = get_intersection_line3D(la_plotly_planes_list[0], sa_plotly_planes_list[j])

                        # print(i, j, la_intsct_line3D[0].points)

                        # error if use sa_intsct_line3D
                        la_intsct_points3D = la_intsct_line3D[0].points
                        # LA 0 SA 2 sa_intsct_points3D : (Point3D(a1, -b1, 0), Point3D(a2, -b2, c2))
                        # LA 0 SA 2 la_intsct_points3D : (Point3D(a1, -b1, 0), Point3D(-a2, b2, -c2))

                        la_intsct_points2D = get_intersection_points2D_fixed(la_intsct_points3D, la_trans_mat2D)
                        la_p1, la_p2 = get_intersection_points2D_with_img(la_intsct_points2D, la_plane_range)
                        la_img = la_plotly_planes_list[0].surfacecolor
                        degree = get_angle(la_p1, la_p2)
                        r_la_img, r_la_p1, r_la_p2 = rotate_image(la_img, degree, la_p1, la_p2)

                        # print('image dimensions  ', la_img.shape, ', after rotation => ', r_la_img.shape)

                        nrow = r_la_img.shape[1]

                        # Making up for abs(degree) in get_angle()
                        # Actually it would be more exact to call it '180_degree_flag'
                        if j == 0:
                            if r_la_p1[1] < nrow/2:
                                rot_flag = 1
                            else:
                                rot_flag = 0

                        if rot_flag == 1:
                            r_la_img, r_la_p1, r_la_p2 = rotate_image(r_la_img, 180.0, r_la_p1, r_la_p2)
                        else:
                            pass

                        la_intersection_points_list[i].append([r_la_p1, r_la_p2])

                        if j == 0:
                            la_img_list.append(r_la_img)

                        estimated_la = get_est_la_plane_from_img_stack(sa_intsct_points2D_with_img, interpolated_img_stack)

                        if FLAG_PRINT_SPECIFICS:
                            print("LA", i, "SA", j)
                            # print("degree :", degree)
                            # print("rot_flag", rot_flag)
                            # print('estimated_la, ', estimated_la.shape)

                        estimated_la_img_list[i].append(estimated_la[::-1])

                    HTML_flag = False
                    if HTML_flag:

                        if dataset=='CAT':
                            html_file_path = pathlib.Path(os.path.join('html_plotly', f"{str(la_file_name.name).split('.')[0]}.html"))
                        elif dataset=='KAG':
                            html_file_path = pathlib.Path(os.path.join('html_plotly', f"{patient}_LA{i+1}.html"))

                        if not html_file_path.exists():
                            Util.plot_planes_with_buttons(sa_plotly_planes_list, la_plotly_planes_list, sa_file_names_list,
                                                          width=1000, height=1000, title="Plotting SA planes with a LA plane", filename=str(html_file_path))
                            print(f"{str(html_file_path)} created")

                        webbrowser.open_new(str(html_file_path))

                    # End of a LA image & corresponding SA images
                # End of all LA images & corresponding SA images

                if FLAG_PRINT_SPECIFICS:
                    # print('la_intersection_points_list, ', len(la_intersection_points_list))
                    print('la_img_list, ', len(la_img_list))
                    print('estimated_la_img_list, ', len(estimated_la_img_list))

                if False:
                    file_slider_fig = FileSlider.FileSliderFig(la_img_list, sa_img_list, estimated_la_img_list,
                                                               la_intersection_points_list, sa_intersection_points_list,
                                                               la_titles_list, sa_titles_list, est_la_titles_list,
                                                               [0.25, 0.95, 0.5, 0.03], [0.25, 0.905, 0.5, 0.03])
                    file_slider_fig.imshow()
                    file_slider_fig.show()

                dat = {}
                dat['la_inter_pts_list'] = la_intersection_points_list
                dat['la_img_list'] = la_img_list
                dat['sa_img_list'] = sa_img_list
                dat['est_la_img_list'] = estimated_la_img_list
                dat['sa_inter_pts_list'] = sa_intersection_points_list

                dir1 = os.path.join(os.getcwd(), save_pkl_dir)

                if dataset=='CAT':
                    fname = os.path.join(dir1, patient_folder_path[-10:]+'_ph='+str(phaseno)+'_la_pair.pkl')
                elif dataset=='KAG':
                    fname = os.path.join(dir1, patient + '_ph='+str(phaseno)+ '_la_pair.pkl')

                if save_pkl:
                    with open(fname, 'wb') as f:
                        pickle.dump(dat, f)

        print("Done!!")


if __name__ == "__main__":
    main()
