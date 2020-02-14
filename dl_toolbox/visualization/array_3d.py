import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from skimage.measure import regionprops

from matplotlib.colors import LinearSegmentedColormap


# Define color map. The custom color map goes from transparent
# black to semi-transparent red and is used as an overlay.
cdict = {'red': [(0.0, 0.0, 0.0),
                (1.0, 0.6, 0.0)],
        'green': [(0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0)],
        'blue': [(0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0)],
        'alpha': [(0.0, 0.0, 0.0),
                (1.0, 0.4, 0.0)]
}
# Create map and register
custom_roi_cmap = LinearSegmentedColormap("custom_roi_cmap", cdict)
plt.register_cmap(cmap=custom_roi_cmap)


def plot_3d_array(ct, seg=None, title="", output_dir=None, ax=None):
    """
    ct: shape (z, y, x, c)
    """
    # to shape (z, y, x, c)
    #ct = np.transpose(ct, axes=(3, 2, 1, 0))
    #if seg is not None:
    #    seg = np.transpose(seg, axes=(3, 2, 1, 0))
    n_slices = len(ct)
    n_rows = int(np.sqrt(n_slices))
    n_cols = n_slices // n_rows
    if n_slices % n_rows > 0:
        n_cols += 1

    if ax is None:
        # we create a new plot
        width = n_cols * 4
        height = n_rows * 4
        f, ax = plt.subplots(n_rows, n_cols, figsize=(width, height), dpi=300,
                             squeeze=False)
    else:
        # we have to check that we have enough ax to plot all slices
        assert ax.shape[0] >= n_rows and ax.shape[1] >= n_cols
        f = ax[0, 0].figure

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= n_slices:
                continue

            ax[i, j].imshow(ct[idx].squeeze(), cmap='gray')
            ax[i, j].set_title(title + "\nSlice {}".format(idx))
            #ax[i, j].axis("off")

            if seg is not None:
                ax[i, j].imshow(
                    seg[idx].squeeze(), cmap="custom_roi_cmap", alpha=0.4)
    # f.tight_layout()

    if output_dir is None:
        # plt.show()
        return f
    else:
        print("stored 3d array plot to {}".format(output_dir))
        f.savefig(output_dir)
