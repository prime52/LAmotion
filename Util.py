import pathlib
import re
import pydicom as dicom
import numpy as np
import plotly
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt


def is_invertible(x: np.ndarray) -> bool:
    return x.shape[0] == x.shape[1] and np.linalg.matrix_rank(x) == x.shape[0]


def sort_by_plane_number(path: pathlib.Path):
    """
    Used as key param for sorted()
    The value of the key parameter should be a function that takes a single argument
    and returns a key to use for sorting purposes
    :param path:
    :return: int
    """
    return int((re.split(r'(\d+)', str(path)))[-4])


def get_pos(dcm: dicom.dataset.FileDataset) -> np.ndarray:
    return np.array([float(p) for p in dcm.ImagePositionPatient])


def get_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray, surfacecolor: np.ndarray,
              colorscale='Greys', showscale: bool = False, reversescale: bool = True) -> plotly.graph_objs.Surface:

    return plotly.graph_objs.Surface(x=x, y=y, z=z, surfacecolor=surfacecolor, cauto=True,
                                     colorscale=colorscale, showscale=showscale, reversescale=reversescale)


def get_plane_xy_range(plane: np.ndarray) -> np.ndarray:
    """
    print("get_plane_xy_range : \n", np.array([[x_min, y_min], [x_max, y_max]], dtype=np.float32))
    get_plane_xy_range : [[  0.   0.]
    [256. 256.]]
    :param plane:
    :return:
    """
    x_max, y_max = plane.shape
    x_min, y_min = 0.0, 0.0

    return np.array([[x_min, y_min], [x_max, y_max]], dtype=np.float32)


def get_plane_z_range(plane: np.ndarray) -> np.ndarray:
    """
    print("get_plane_z_range : \n", np.array([[z_min, z_min], [z_max1, z_max2]], dtype=np.float32))
    get_plane_z_range : [[  0.   0.]
    [256. 256.]]
    :param plane:
    :return:
    """
    z_max1, z_max2 = plane.shape
    z_min = 0.0
    return np.array([[z_min, z_min], [z_max1, z_max2]], dtype=np.float32)


def get_dicom_file(file_name: pathlib.Path) -> dicom.dataset.FileDataset:
    dicom_file = dicom.read_file(str(file_name))
    return dicom_file


# Not Used

# Suppose that the intersection of two planes is a line
def smart_crop2D(img: np.ndarray, threshold=0.0) -> np.ndarray:
    non_empty_columns = np.where(img.max(axis=0) > threshold)[0]
    non_empty_rows = np.where(img.max(axis=1) > threshold)[0]
    crop_box = (np.min(non_empty_rows), np.max(non_empty_rows),
                np.min(non_empty_columns), np.max(non_empty_columns))
    new_img = np.copy(img[crop_box[0]:crop_box[1] + 1, crop_box[2]:crop_box[3] + 1, ...])
    return new_img


def get_n_points_from_img(n: int, img: np.ndarray, cmap='gray') -> list:
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(img, cmap=cmap)
    points_list = plt.ginput(n, timeout=-1)
    plt.close(fig)
    return points_list


def get_update_menus(sa_plotly_planes_list: list, sa_file_names: list) -> list:
    def get_visibility(i):
        return [x for x in [True if i == j else False for j in range(1, len(sa_plotly_planes_list) + 1)] + [True]]

    def get_plane_buttons():
        plane_buttons = [dict(label="SA" + (re.split(r'(\d+)', str(sa_file_names[i - 1])))[-4],
                              method='update', args=[{'visible': get_visibility(i)}])
                         for i in range(1, len(sa_plotly_planes_list) + 1)]
        return plane_buttons

    n_sa_planes = len(sa_plotly_planes_list)
    update_menus = list([
        dict(type="buttons",
             active=-1,
             buttons=list([
                 dict(label='LA ONLY',
                      method='update',
                      args=[{'visible': [False] * n_sa_planes + [True]}]),
                 dict(label='RESET',
                      method='update',
                      args=[{'visible': [True] * n_sa_planes + [True]}])
             ]) + get_plane_buttons()
             )
    ])
    return update_menus


def plot_planes(planes_list: list, width: int = 1000, height: int = 1000,
                title: str = 'plotly') -> None:
    layout = dict(width=width, height=height, title=title)
    fig = plotly.graph_objs.Figure(data=planes_list, layout=layout)
    plotly.offline.iplot(fig)


def plot_planes_with_buttons(sa_plotly_planes_list: list, la_plotly_planes_list: list, sa_file_names: list,
                             width: int = 1000, height: int = 1000,
                             title: str = 'plotly', filename: str = None) -> None:
    update_menus = get_update_menus(sa_plotly_planes_list, sa_file_names)
    layout = dict(width=width, height=height, title=title, updatemenus=update_menus)
    fig = plotly.graph_objs.Figure(data=sa_plotly_planes_list + la_plotly_planes_list, layout=layout)
    if filename:
        plotly.offline.plot(fig, filename=filename, auto_open=False)
    else:
        plotly.offline.iplot(fig)

