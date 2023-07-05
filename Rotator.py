import matplotlib as mpl
mpl.use('Qt5Agg')

from matplotlib.widgets import Slider, Button
import numpy as np
from skimage.transform import rotate


class RotatableAxes:
    def __init__(self, fig: mpl.figure.Figure, axes: mpl.axes.Axes,
                 rect_angle: list, rect_reset: list):
        self.fig = fig
        # Suppose that there exists an image in the axes
        self.axes = axes
        self.renderer = self.axes.figure.canvas.get_renderer()
        self.axes_img = self.axes.get_images()[0]
        self.original_axes_img = self.axes_img
        self.original_img_list = [[np.rot90(img, i) for i in range(4)] for img in [self.axes_img._A, self.axes_img._A[::-1, :]]]
        self.rot_idx = 0
        self.flip_idx = 0
        self.axes_for_angle_slider = self.fig.add_axes(rect_angle)
        self.axes_for_reset_button = self.fig.add_axes(rect_reset)
        self.angle_slider = Slider(self.axes_for_angle_slider, 'Angle(Degree)', 0.0,
                                   359.9, valinit=0.0, valstep=0.1)
        self.angle_slider.on_changed(self.update_img)
        self.reset_button = Button(self.axes_for_reset_button, 'Reset')
        self.reset_button.on_clicked(self.reset)

    def connect(self) -> None:
        # connect to all the events we need
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def disconnect(self) -> None:
        # disconnect all the stored connection ids
        self.fig.canvas.mpl_disconnect(self.onclick)

    def update_la_img(self) -> None:
        self.axes_img = self.axes.get_images()[0]
        self.original_axes_img = self.axes_img
        self.original_img_list = [[np.rot90(img, i) for i in range(4)] for img in [self.axes_img._A, self.axes_img._A[::-1, :]]]
        self.rot_idx = 0
        self.flip_idx = 0
        self.angle_slider.reset()

    def update_after_rot90(self) -> None:
        self.rot_idx = (self.rot_idx + 1) % 4
        left, right, bottom, top = self.original_axes_img.get_extent()
        self.axes_img.set_extent([top, bottom, right, left])

    def update_after_flip(self) -> None:
        self.flip_idx = (self.flip_idx + 1) % 2

    def onclick(self, event: mpl.backend_bases.Event) -> None:
        if self.axes == event.inaxes:
            if event.button == mpl.backend_bases.MouseButton.LEFT:
                self.update_after_rot90()
            elif event.button == mpl.backend_bases.MouseButton.RIGHT:
                self.update_after_flip()
            self.angle_slider.set_val(self.angle_slider.val)
            self.axes.figure.canvas.draw()
            self.axes.figure.canvas.flush_events()

    def update_img(self, new_angle: float) -> None:
        rotated_img = rotate(self.original_img_list[self.flip_idx][self.rot_idx], new_angle)
        self.axes_img.set_data(rotated_img)
        self.axes.figure.canvas.update()
        self.axes.figure.canvas.flush_events()

    def reset(self, event: mpl.backend_bases.Event) -> None:
        self.angle_slider.reset()
