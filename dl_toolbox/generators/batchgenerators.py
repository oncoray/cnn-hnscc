import numpy as np
import os

from keras.preprocessing.image import Iterator
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase


class DataLoader(SlimDataLoaderBase):
    def __init__(self, data, batch_size,
                 labels=None):
        """
        Parameters
        -----------
        data: np.array (n, z, y, x, c)
        labels: np.array with len(labels) == len(data) or None
        """
        super().__init__(data, batch_size)
        # data is now stored in self._data.
        self._labels = labels

        # we need to set self.indices manually
        self.indices = None

    def generate_train_batch(self):
        # the interface function required by batchgenerators API
        if self.indices is None:
            raise ValueError(
                "self.indices is None. Make sure to set it before calling this function!")

        pat_idx = self.indices
        img_batch = [None] * len(pat_idx)
        for i, idx in enumerate(pat_idx):
            img = self._data[idx]
            # reshape from (z, y, x, c) to (c, x, y, z) or 2D respectively
            axes = list(range(img.ndim))  # [0, 1, 2,..., ndim-1]
            axes_transpose = axes[::-1]
            img_batch[i] = np.transpose(img, axes=axes_transpose)

        # (b, c, x, y, z) or 2D respecively
        img_batch = np.array(img_batch)

        # now construct the dictionary and return it. keys 'data' and 'seg' are treated
        # specially by the batchgenerators API to compute same tranform on both
        ret_val = {'data': img_batch, 'idx': pat_idx}

        if self._labels is not None:
            label_batch = [None] * len(pat_idx)
            for i, idx in enumerate(pat_idx):
                label_batch[i] = self._labels[idx]

            ret_val['labels'] = np.array(label_batch)

        return ret_val

    # this should now be used as interface function
    def generate_batch(self, index_array):
        self.indices = index_array
        return self.generate_train_batch()


class NumpyArrayIteratorUpTo3D(Iterator):
    """This uses the batchgenerators API and allows 3D data augmentation with keras."""

    def __init__(self,
                 # (n_samples, z, y, x, channels) for 3D or (n_samples, y, x, channels) for 2D data
                 x,
                 y,
                 transform,  # the transformations from the batchgenerators API
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):

        self.x = self._check_and_transform_x(x)

        self.y = self._check_and_transform_y(y)

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        super().__init__(x.shape[0],
                         batch_size,
                         shuffle,
                         seed)

        if not isinstance(transform, Compose):
            if isinstance(transform, list):
                self.transform = Compose(transform)
            else:
                self.transform = Compose([transform])
        else:
            self.transform = transform

        # print("NumpyArrayIteratorUpTo3D: self.transform=", self.transform)

        self.data_loader = self._create_data_loader()

    def _create_data_loader(self):
        return DataLoader(
            data=self.x, labels=self.y, batch_size=self.batch_size)

    def _check_and_transform_x(self, x):
        # supports samples of 2D and 3D (with color channel),
        # i.e. (n, z, y, x, c) or (n, y, x, c)
        assert x.ndim in [4, 5]

        return np.asarray(x)

    def _check_and_transform_y(self, y):
        if y is not None and len(self.x) != len(y):
            raise ValueError(
                "`y` does not have same length as x!"
                f"Found: len(y)={len(y)}, len(x)={len(self.x)}")

        if y is not None:
            y = np.asarray(y)

        return y

    def _get_batches_of_transformed_samples(self, index_array):
        item = self.data_loader.generate_batch(index_array)
        assert np.all(index_array == item["idx"])

        transformed = self.transform(**item)

        labels = transformed.get("labels")  # might be None

        # is still in format (b, c, x, y, z) or 2d respectively
        imgs = transformed["data"]
        # (0, 1, 2, 3) for 2D or (0, 1, 2, 3, 4) for 3D with batches and channels first
        axes = list(range(imgs.ndim))
        # now (0, 3, 2, 1) or (0, 4, 3, 2, 1) which would be (b, z, y, x, c)
        transpose_axes = [axes[0]] + axes[1:][::-1]
        imgs = np.transpose(imgs, axes=transpose_axes)

        # if self.save_to_dir:
        #     os.makedirs(self.save_to_dir, exist_ok=True)
        #     for i, j in enumerate(index_array):
        #         img = imgs[i]
        #         fname = '{prefix}_index{index}_{hash}.{format}'.format(
        #             prefix=self.save_prefix,
        #             index=j,
        #             hash=np.random.randint(1e4),
        #             format=self.save_format)

        #         plot_3d_array(
        #             img, title=f"Sample {j}",
        #             output_dir=os.path.join(self.save_to_dir, fname))

        if labels is None:
            return imgs
        else:
            return imgs, labels

    def get_data_shapes(self):
        return [self.x.shape[1:]]
