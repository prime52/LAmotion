import matplotlib as mpl
mpl.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.widgets import Slider, Button

from skimage import exposure
import os

import numpy as np
import plotly

import Rotator

plotly.offline.init_notebook_mode(connected=True)


class FileSliderFig:

    def __init__(self, la_img_list: list,
                 sa_img_list: list,
                 est_la_img_list: list,
                 la_intersection_points_list: list,
                 sa_intersection_points_list: list,
                 la_titles_list: list,
                 sa_titles_list: list,
                 est_la_titles_list: list,
                 la_rect: list,
                 sa_rect: list):
        fig, axes_list = plt.subplots(1, 3, tight_layout=False)
        self.fig = fig
        self.la_axes = axes_list[0]
        self.sa_axes = axes_list[1]
        self.est_la_axes = axes_list[2]
        self.la_img_list = la_img_list
        self.sa_img_list = sa_img_list
        self.est_la_img_list = est_la_img_list
        self.la_intersection_points_list = la_intersection_points_list
        self.sa_intersection_points_list = sa_intersection_points_list
        self.la_titles_list = la_titles_list
        self.sa_titles_list = sa_titles_list
        self.est_la_titles_list = est_la_titles_list
        self.n_la_images = len(self.la_img_list)
        self.n_sa_images = len(self.sa_img_list)
        self.axes_for_la_file_slider = self.fig.add_axes(la_rect)
        self.axes_for_sa_file_slider = self.fig.add_axes(sa_rect)
        self.la_file_slider = Slider(self.axes_for_la_file_slider, 'LA INDEX',
                                     1.0, self.n_la_images, valinit=1.0, valstep=1.0)
        self.sa_file_slider = Slider(self.axes_for_sa_file_slider, 'SA INDEX',
                                     1.0, self.n_sa_images, valinit=1.0, valstep=1.0)
        self.la_file_slider.on_changed(self.update_la_file)
        self.sa_file_slider.on_changed(self.update_sa_file)
        self.rot_axes = None

    def imshow(self) -> None:
        la_img = self.la_img_list[0]
        sa_img = self.sa_img_list[0]

        p2, p98 = np.percentile(la_img, (2, 98))
        print(f'la_img max={np.max(la_img)}, p2={p2}, p98={p98}')
        la_img = exposure.rescale_intensity(la_img, in_range=(p2, p98), out_range=(p2, p98))
        # la_img = exposure.equalize_adapthist(la_img, clip_limit=0.03)
        print(f'la_img max = {np.max(la_img)} ')

        # self.la_axes.imshow(la_img, cmap='gray', clim=(0, 1.0)) #, vmin=0.1*np.max(la_img), vmax=0.3*np.max(la_img))
        self.la_axes.imshow(la_img, cmap='gray')
        self.sa_axes.imshow(sa_img, cmap='gray')
        self.est_la_axes.imshow(self.est_la_img_list[0][0], cmap='gray')

        sa_p1, sa_p2 = self.sa_intersection_points_list[0][0]
        self.sa_axes.plot((sa_p1[0], sa_p2[0]), (sa_p1[1], sa_p2[1]), 'r--')

        la_p1, la_p2 = self.la_intersection_points_list[0][0]
        self.la_axes.plot((la_p1[0], la_p2[0]), (la_p1[1], la_p2[1]), 'r--')

        set_axes_extent(la_img, self.la_axes)
        set_axes_extent(sa_img, self.sa_axes)
        set_axes_extent(self.est_la_img_list[0][0], self.est_la_axes)

        self.la_axes.title.set_text(self.la_titles_list[0])
        self.sa_axes.title.set_text(self.sa_titles_list[0])
        self.est_la_axes.title.set_text(self.est_la_titles_list[0])

        self.rot_axes = Rotator.RotatableAxes(self.fig, self.la_axes,
                                 [0.25, 0.06, 0.5, 0.03], [0.72, 0.01, 0.03, 0.03])
        self.rot_axes.connect()

    def show(self) -> None:
        self.fig.canvas.manager.window.showMaximized()
        plt.show()

    def update_la_file(self, new_la_slider_val: float) -> None:
        new_la_idx = int(new_la_slider_val - 1.0)
        cur_sa_idx = int(self.sa_file_slider.val - 1.0)

        set_axes_img(self.la_img_list[new_la_idx], self.la_axes)

        folder1, fname1 = os.path.split(self.la_titles_list[new_la_idx])

        self.la_axes.set_title(f'{folder1}\n {fname1}')
        self.la_axes.figure.canvas.update()
        self.la_axes.figure.canvas.flush_events()

        self.rot_axes.update_la_img()

        sa_axes_lines_list = self.sa_axes.get_lines()
        sa_p1, sa_p2 = self.sa_intersection_points_list[new_la_idx][cur_sa_idx]
        sa_axes_lines_list[0].set_data((sa_p1[0], sa_p2[0]), (sa_p1[1], sa_p2[1]))
        self.sa_axes.figure.canvas.update()
        self.sa_axes.figure.canvas.flush_events()

        la_axes_lines_list = self.la_axes.get_lines()
        la_p1, la_p2 = self.la_intersection_points_list[new_la_idx][cur_sa_idx]
        la_axes_lines_list[0].set_data((la_p1[0], la_p2[0]), (la_p1[1], la_p2[1]))
        self.la_axes.figure.canvas.update()
        self.la_axes.figure.canvas.flush_events()

        set_axes_img(self.est_la_img_list[new_la_idx][cur_sa_idx], self.est_la_axes)


        self.est_la_axes.set_title(self.est_la_titles_list[new_la_idx])
        self.est_la_axes.figure.canvas.update()
        self.est_la_axes.figure.canvas.flush_events()
        self.fig.canvas.draw()

    def update_sa_file(self, new_sa_slider_val: float) -> None:
        cur_la_idx = int(self.la_file_slider.val - 1.0)
        new_sa_idx = int(new_sa_slider_val - 1.0)

        set_axes_img(self.sa_img_list[new_sa_idx], self.sa_axes)

        sa_axes_lines_list = self.sa_axes.get_lines()
        sa_p1, sa_p2 = self.sa_intersection_points_list[cur_la_idx][new_sa_idx]
        sa_axes_lines_list[0].set_data((sa_p1[0], sa_p2[0]), (sa_p1[1], sa_p2[1]))

        folder1, fname1 = os.path.split(self.sa_titles_list[new_sa_idx])

        self.sa_axes.set_title(f'{folder1}\n {fname1}')
        self.sa_axes.figure.canvas.update()
        self.sa_axes.figure.canvas.flush_events()

        la_axes_lines_list = self.la_axes.get_lines()
        la_p1, la_p2 = self.la_intersection_points_list[cur_la_idx][new_sa_idx]
        la_axes_lines_list[0].set_data((la_p1[0], la_p2[0]), (la_p1[1], la_p2[1]))

        folder2, fname2 = os.path.split(self.la_titles_list[cur_la_idx])

        self.la_axes.set_title(f'{folder2}\n{fname2}')
        self.la_axes.figure.canvas.update()
        self.la_axes.figure.canvas.flush_events()

        set_axes_img(self.est_la_img_list[cur_la_idx][new_sa_idx], self.est_la_axes)
        self.est_la_axes.figure.canvas.update()
        self.est_la_axes.figure.canvas.flush_events()
        self.fig.canvas.draw()


def set_axes_extent(new_img: np.ndarray, axes: mpl.axes.Axes) -> None:
    axes_img = axes.get_images()[0]

    n_row, n_col = new_img.shape
    axes_img.set_extent([0, n_col, n_row, 0])


def set_axes_img(new_img: np.ndarray, axes: mpl.axes.Axes) -> None:
    axes_img = axes.get_images()[0]

    axes_img.set_data(new_img)

    n_row, n_col = new_img.shape
    axes_img.set_extent([0, n_col, n_row, 0])
