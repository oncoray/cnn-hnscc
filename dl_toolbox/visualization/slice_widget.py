from ipywidgets import widgets
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import collections


class SliceVisualizer(widgets.VBox):
    """This widget works only for matplotlib widget mode within jupyter notebooks."""
    def __init__(self, data, roi=None, bbox_center=None, bbox_dims=(1, 1, 1), cmap="gray"):
        """
        Parameters
        ----------
        data: 4D np.array (samples x height x width x channels)
        bbox_center: tuple of length 3 of type int
            specify the zyx indices of the lower left of the bounding box to plot
        bbox_dims: tuple of int of length 3 for extent in z, y and x direction
        """
        super().__init__()
        self.cmap = cmap
        self.data = data

        self.disp_data = data

        # we don't allow to switch the last axis which are channels
        self.coord = widgets.Dropdown(description="slice axis",
                                      # don't allow changing of directions for now
                                      # since bounding box plotting does not work with that
                                      options=[0],
                                      #options=list(range(len(self.data.shape)-1)),
                                      value=0)
        self.coord.observe(self.reshape_disp_data, names='value')

        self.format_label = widgets.Label(str(self.data.shape))

        self.slider = widgets.IntSlider(description="Slice",
                                        min=0,
                                        max=self.data.shape[0]-1,
                                        value=0,
                                        continuous_update=False)
        self.slider.observe(self.plot_slice, names='value')

        self.roi_data = roi
        n = 1 if self.roi_data is None else 2
        self.fig, self.ax = plt.subplots(1, n, figsize=(10, 5))
        if not isinstance(self.ax, collections.Iterable):
            self.ax = [self.ax]

        self.plt_box = widgets.HBox([self.fig.canvas])
        self.children = [
            widgets.HBox([self.coord, self.format_label]),
            self.slider,
            self.plt_box]

        self.bboxes = None
        self.bbox_dims = bbox_dims
        self.bbox_center = bbox_center
        self.bbox_center_patches = None

        self.roi_patch = None

        if self.bbox_center is not None:
            self.bboxes = [patches.Rectangle(
                xy=(self.bbox_center[2] - self.bbox_dims[2] // 2, self.bbox_center[1] - self.bbox_dims[1] // 2),
                width=bbox_dims[2], height=bbox_dims[1],
                linewidth=1, edgecolor='r', facecolor='none') for i in range(n)]
            self.bbox_center_patches = [patches.Circle(
                xy=(self.bbox_center[2], self.bbox_center[1]), radius=1,
                edgecolor='r', fill=False) for i in range(n)]

            for i in range(len(self.ax)):
                self.ax[i].add_patch(self.bboxes[i])
                self.ax[i].add_patch(self.bbox_center_patches[i])

        self._plot_slice(0)

    def _reshape_disp_data(self, new_front_idx):
        """
        Moves given index dimension to the front.
        Always based on originally read data.
        """
        # move selected axis to front
        # TODO: think if np.moveaxis is the right thing to do
        self.disp_data = np.moveaxis(self.data, new_front_idx, 0)

        # update the format label
        self.format_label.value = str(self.disp_data.shape)

        # update the slider to the amount of available data
        self.slider.max = self.disp_data.shape[0] - 1

        # clear the axis
        self._clear_plot()

        print(self.disp_data.shape)

    def reshape_disp_data(self, change):
        #print("change", change)
        self._reshape_disp_data(int(change['new']))
        print("reshape_disp_data from", self.data.shape, "to", self.disp_data.shape)

    def _plot_slice(self, idx):
        self.ax[0].imshow(self.disp_data[idx].squeeze(), cmap=self.cmap)
        if self.roi_data is not None:
            self.ax[1].imshow(self.roi_data[idx].squeeze(), cmap=self.cmap)

            if self.roi_patch is not None:
                # TODO: maybe we need to get rid of the old patch
                #pass
                try:
                    self.roi_patch.remove()
                except:
                    print("Removing self.roi_patch failed!")

            roi_props = regionprops(self.roi_data[idx].squeeze())
            if roi_props:
                coords = roi_props[0].coords
                # need to swap columns
                coords[:, [0, 1]] = coords[:, [1, 0]]
                self.roi_patch = patches.Polygon(
                    coords, edgecolor='r', fill=True, alpha=0.3)
                self.ax[0].add_patch(self.roi_patch)

        if self.bboxes is not None:
            # check if the selected slice is in the bounding box range before plotting
            slice_idx = self.slider.value
            r = int(self.bbox_dims[0] // 2)
            box_limits = (self.bbox_center[0] - r, self.bbox_center[0] + r)
            if box_limits[0] < slice_idx and slice_idx < box_limits[1]:
                for box in self.bboxes:
                    box.set_visible(True)
                for center in self.bbox_center_patches:
                    center.set_visible(True)
            else:
                for box in self.bboxes:
                    box.set_visible(False)
                for center in self.bbox_center_patches:
                    center.set_visible(False)

        self._update_plot()

    def plot_slice(self, change):
        #print("plot_slice", change['new'])
        self._plot_slice(int(change['new']))

    def _clear_plot(self):
        self.ax.clear()
        self._update_plot()

    def _update_plot(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()