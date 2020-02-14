"""
classes for usage within jupyter notebooks.
this requires the following code snippet to be
executed as first notebook cell

%matplotlib widget
import matplotlib.pyplot as plt
plt.ioff() # deactivate instant plotting is necessary!
"""

from ipywidgets import widgets
from functools import partial
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import os

from .array_3d import plot_3d_array


class AugmentationVisualizer(widgets.HBox):
    def __init__(self, data_handler, output_dir, scale=1):
        super().__init__()

        assert data_handler.data_ready
        self.data_handler = data_handler

        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # always use batch size of 1
        self.data_handler.batch_size = 1
        # this is the number of 2d plots we need to draw
        n_slices = self.data_handler.crop_size[0]
        self.n_rows = int(np.sqrt(n_slices))
        self.n_cols = n_slices // self.n_rows
        if n_slices % self.n_rows > 0:
            self.n_cols += 1
        width = self.n_cols * scale
        height = self.n_rows * scale
        self.fig_ori, self.ax_ori = plt.subplots(
            self.n_rows, self.n_cols, figsize=(width, height), squeeze=False)
        self.fig_aug, self.ax_aug = plt.subplots(
            self.n_rows, self.n_cols, figsize=(width, height), squeeze=False)

        self.fig_ori.tight_layout()
        self.fig_aug.tight_layout()

        for axes in [self.ax_ori, self.ax_aug]:
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    axes[i, j].axis("off")
                    # axes[i, j].set_title(f"Slice {i*n_cols + j}")

        # "do_rotation" true/false
        self.do_rotation = widgets.Checkbox(
            value=True,
            description='do_rotation',
            disabled=False,
            indent=False)
        # p_rot_per_sample float 0.5
        self.p_rot_per_sample = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            description="p_rot_per_sample",
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f')
        # rotation angles
        self.augment_slider_rot = widgets.FloatSlider(
            value=15.,
            min=0.,
            max=360,
            description="rotation angle")

        # "do_elastic_deform" true/false
        self.do_elastic_deform = widgets.Checkbox(
            value=True,
            description='do_elastic_deform',
            disabled=False,
            indent=False)
        # p_el_per_sample float 0.5
        self.p_el_per_sample = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            description="p_el_per_sample",
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f')
        # "deformation_scale"  tuple (0, 0.25)
        self.deformation_scale = widgets.FloatRangeSlider(
            value=[0, 0.25],
            min=0,
            max=1.0,
            step=0.05,
            description='deformation_scale',
            # disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f')

        # "do_scale" true/false
        self.do_scale = widgets.Checkbox(
            value=True,
            description='do_scale',
            disabled=False,
            indent=False)
        # p_scale_per_sample float 0.5
        self.p_scale_per_sample = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            description="p_scale_per_sample",
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f')
        # "scale" tuple (0.75, 1.25)
        self.scale = widgets.FloatRangeSlider(
            value=[0.75, 1.25],
            min=0,
            max=5.0,
            step=0.1,
            description='scale',
            # disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f')
        # "do_mirror", true/false
        self.do_mirror = widgets.Checkbox(
            value=True,
            description='do_mirror',
            disabled=False,
            indent=False)
        # "p_per_sample" float 0.15
        # for gamma, gaussian noise, brightness,
        self.p_per_sample = widgets.FloatSlider(
            value=0.15,
            min=0,
            max=1.0,
            description="p_per_sample",
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f')
        # "brightness_range" tuple (0.7, 1.5)
        self.brightness_range = widgets.FloatRangeSlider(
            value=[0.7, 1.5],
            min=0,
            max=5.0,
            step=0.1,
            description='brightness_range',
            # disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f')
        # "gaussian_noise_variance" tuple (0, 0.05)
        self.gaussian_noise_variance = widgets.FloatRangeSlider(
            value=[0., 0.05],
            min=0,
            max=1.0,
            step=0.05,
            description='gaussian_noise_variance',
            # disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f')
        # "gamma_range" tuple (0.5, 2)
        self.gamma_range = widgets.FloatRangeSlider(
            value=[0.5, 2.0],
            min=0,
            max=5.0,
            step=0.1,
            description='gamma_range',
            # disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f')


        self.arg_to_widget_map = {
            # rotations
            "do_rotation": self.do_rotation,
            "p_rot_per_sample": self.p_rot_per_sample,
            "angle_x": self.augment_slider_rot,
            "angle_y": self.augment_slider_rot,
            "angle_z": self.augment_slider_rot,
            # elastic deforms
            "do_elastic_deform": self.do_elastic_deform,
            "p_el_per_sample": self.p_el_per_sample,
            "deformation_scale": self.deformation_scale,
            # scaling
            "do_scale": self.do_scale,
            "p_scale_per_sample": self.p_scale_per_sample,
            "scale": self.scale,
            # mirroring
            "do_mirror": self.do_mirror,
            # all others
            "p_per_sample": self.p_per_sample,
            "brightness_range": self.brightness_range,
            "gaussian_noise_variance": self.gaussian_noise_variance,
            "gamma_range": self.gamma_range,
        }

        self.patient_dropdown = widgets.Dropdown(
            options=self.data_handler.patient_ids, value=None)
        self.patient_dropdown.observe(self._init_patient_cb, names="value")

        self.apply_aug_button = widgets.Button(
            description="Augment", button_style="success")
        self.apply_aug_button.on_click(self._plot_augmented_batch_cb)

        self.output_widget = widgets.Output(layout={'border': '1px solid black'})

        self.children = [
            widgets.VBox(children=[
                self.patient_dropdown,
                widgets.HBox(children=[
                    self.fig_ori.canvas,
                    widgets.VBox(children=[
                        widgets.Label("Augmentation args"),
                        # rotation children
                        widgets.VBox(children=[
                            self.do_rotation,
                            self.p_rot_per_sample,
                            self.augment_slider_rot,
                        ], layout={'border': '1px solid black'}),
                        # elastic deform children
                        widgets.VBox(children=[
                            self.do_elastic_deform,
                            self.p_el_per_sample,
                            self.deformation_scale,
                        ], layout={'border': '1px solid black'}),
                        # scaling
                        widgets.VBox(children=[
                            self.do_scale,
                            self.p_scale_per_sample,
                            self.scale,
                        ], layout={'border': '1px solid black'}),
                        # mirroring
                        widgets.VBox(children=[
                            self.do_mirror,
                        ], layout={'border': '1px solid black'}),
                        # all others
                        widgets.VBox(children=[
                            self.p_per_sample,
                            self.brightness_range,
                            self.gaussian_noise_variance,
                            self.gamma_range
                        ], layout={'border': '1px solid black'})
                    ]),
                    self.fig_aug.canvas,
                ]),
                self.apply_aug_button,
                self.output_widget]),
        ]

        self.generator = None

        self.output_widget.clear_output()
        with self.output_widget:
            print(self.ax_ori.shape, self.ax_aug.shape)

    def _init_patient_cb(self, change):
        self.output_widget.clear_output()
        with self.output_widget:
            pat = change["new"]
            #print("will init new patient", pat)

            # create a new generator
            self.generator, _ = self.data_handler.create_generators(
                training_ids=[pat], validation_ids=[pat])

            # clear the old plots of the augmented data
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    self.ax_aug[i, j].clear()
                    self.ax_aug[i, j].axis("off")

            # plot the unaugmented samples
            ori_img = self.data_handler.img_dict[pat]["img"][0]
            #print(ori_img.shape)
            plot_3d_array(
                ori_img, ax=self.ax_ori, title="Original",
                output_dir=os.path.join(self.output_dir, "ori_3d.png"))
            # since interactive mode should be turned off, we have
            # to update the figure explicitely
            self._refresh_plots()

    def _plot_augmented_batch_cb(self, b):
        self.output_widget.clear_output()
        with self.output_widget:
            assert self.generator is not None, "No patient seems to be selected"

            print("plotting augmented batch")

            # TODO: create new generator with the selected augmentation parameters
            for k, widget in self.arg_to_widget_map.items():
                self.data_handler.training_augmentation_args[k] = widget.value

            pat = self.patient_dropdown.value
            if pat is not None:
                self.generator, _ = self.data_handler.create_generators(
                    training_ids=[pat], validation_ids=[pat])

            # create new batch with the given parameters
            images, _ = next(self.generator)
            aug_img = images[0]
            print(aug_img.shape)#, labels.shape)

            # plot the batch into the augmented figures
            plot_3d_array(
                aug_img, ax=self.ax_aug, title="Augmented",
                output_dir=os.path.join(self.output_dir, "aug_3d.png"))
            # since interactive mode should be turned off, we have
            # to update the figure explicitely
            self._refresh_plots()

    def _refresh_plots(self):
        try:
            self.fig_ori.canvas.draw()
            self.fig_ori.canvas.flush_events()
        except ValueError as e:
            warn("{}: drawing the original plot failed! Reason: {}".format(
                type(self), e))

        try:
            self.fig_aug.canvas.draw()
            self.fig_aug.canvas.flush_events()
        except ValueError as e:
            warn("{}: drawing the augmented plot failed! Reason: {}".format(
                type(self), e))
