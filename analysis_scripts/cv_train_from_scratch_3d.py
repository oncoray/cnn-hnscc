import matplotlib
matplotlib.use("Agg")

import argparse
import numpy as np
import os
import pandas as pd

from functools import partial
from keras import optimizers
from keras.preprocessing.image import Iterator
from keras import callbacks as keras_cb

from dl_toolbox.cross_validation.cmd_line_parser import SurvivalCVCmdLineParserBase
from dl_toolbox.models.hosny_publication import hosny_model_3d
from dl_toolbox.data.data_handler import DataHandlerBase
from dl_toolbox.data.read import read_patient_data
from dl_toolbox.preprocessing.preprocessing import crop_3d_around_tumor_center
from dl_toolbox.cross_validation.run_cv_cmd import run_cv_from_cmd
from dl_toolbox.visualization.array_3d import plot_3d_array
from dl_toolbox.callbacks.concordance_index import ConcordanceIndex
from dl_toolbox.losses import neg_cox_log_likelihood

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform,\
    BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from cv_train_from_scratch import FromScratchCVContext


class Range(object):
    """Allows to specify intervals as choices for argparse"""
    # taken from https://stackoverflow.com/a/58004976
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self


def add_data_augmentation_args_batchgenerators(parser_group):

    # rotation
    parser_group.add_argument(
        "--do_rotation", action="store_true", default=False)
    parser_group.add_argument(
        "--p_rot_per_sample", type=float, default=0.,
        choices=Range(0.0, 1.0))
    parser_group.add_argument(
        "--angle_x", type=float, default=0., choices=Range(0.0, 360.0))
    parser_group.add_argument(
        "--angle_y", type=float, default=0., choices=Range(0.0, 360.0))
    parser_group.add_argument(
        "--angle_z", type=float, default=0., choices=Range(0.0, 360.0))

    # elastic deformation
    parser_group.add_argument(
        "--do_elastic_deform", action='store_true', default=False)
    parser_group.add_argument(
        "--p_el_per_sample", type=float, default=0.,
        choices=Range(0.0, 1.0))
    parser_group.add_argument(
        "--deformation_scale", type=float, nargs=2,
        metavar=('lower', 'upper'),
        default=[0., 0.])

    # scaling
    parser_group.add_argument("--do_scale", action="store_true", default=False)
    parser_group.add_argument(
        "--p_scale_per_sample", type=float, default=0.,
        choices=Range(0.0, 1.0))
    parser_group.add_argument(
        "--scale", type=float, nargs=2,
        metavar=('lower', 'upper'),
        default=[0., 0.])

    # mirroring
    parser_group.add_argument("--do_mirror", action="store_true", default=False)

    # probability for all other transforms
    parser_group.add_argument(
        "--p_per_sample", type=float, default=0.,
        choices=Range(0.0, 1.0),
        help="Augmentation probability for BrightnessMultiplicativeTransform,")
             #" GaussianNoiseTransform and GammaTransform")
    parser_group.add_argument(
        "--brightness_range", type=float, nargs=2,
        metavar=('lower', 'upper'),
        default=[0., 0.])
    parser_group.add_argument(
        "--gaussian_noise_variance", type=float, nargs=2,
        metavar=('lower', 'upper'),
        default=[0., 0.])
    parser_group.add_argument(
        "--gamma_range", type=float, nargs=2,
        metavar=('lower', 'upper'),
        default=[0., 0.])


def get_data_augmentation_args_batchgenerators(parsed_args):

    return {'do_rotation': parsed_args.do_rotation,
            'p_rot_per_sample': parsed_args.p_rot_per_sample,
            'angle_x': parsed_args.angle_x,
            'angle_y': parsed_args.angle_y,
            'angle_z': parsed_args.angle_z,
            'do_elastic_deform': parsed_args.do_elastic_deform,
            'p_el_per_sample': parsed_args.p_el_per_sample,
            'deformation_scale': parsed_args.deformation_scale,
            'do_scale': parsed_args.do_scale,
            'p_scale_per_sample': parsed_args.p_scale_per_sample,
            'scale': parsed_args.scale,
            'do_mirror': parsed_args.do_mirror,
            'p_per_sample': parsed_args.p_per_sample,
            'brightness_range': parsed_args.brightness_range,
            'gaussian_noise_variance': parsed_args.gaussian_noise_variance,
            'gamma_range': parsed_args.gamma_range}


class SurvivalCVCmdLineParserDataAug(SurvivalCVCmdLineParserBase):
    """Replace the standard data augmentation command line options."""

    def add_data_handler_args(self):
        super().add_data_handler_args()

        group = self._get_or_create_group("Preprocessing")
        group.add_argument(
            "--crop_size", type=int, nargs=3, default=[32, 64, 64],
            help="Size of each 3D crop of a patient, extracted from the part of the CT scan where the tumor is visible.")
        group.add_argument(
            "--n_random_crops_per_patient", type=int, default=0,
            help="The number of random crops that are used from each patient additional to the central crop around the tumor center of mass.")
        group.add_argument(
            "--max_pixels_shift", type=int, nargs=3, default=[16, 16, 16],
            help="Definition of radii along z,y and x axis from the tumor center of mass from which random crop centers will be drawn."
                 " Only used with n_random_crops_per_patient > 0.")

    def get_data_handler_args(self):
        args = super().get_data_handler_args()

        args["crop_size"] = self.parsed_args.crop_size

        n_random = self.parsed_args.n_random_crops_per_patient
        if n_random < 0:
            n_random = 0
            print["[W]: negative n_crops_per_patient was used. Set it to 0!"]
        args["n_random_crops_per_patient"] = n_random

        pixels_shift = self.parsed_args.max_pixels_shift
        for i, p in enumerate(pixels_shift):
            if p < 0:
                print(f"[W]: max_pixels_shift[{i}] < 0. Set it to 0!")
                pixels_shift[i] = 0
        args["max_pixels_shift"] = pixels_shift

        return args

    def _add_data_augmentation_args(self, parser_group):
        add_data_augmentation_args_batchgenerators(parser_group)

    def _get_data_augmentation_args(self):
        return get_data_augmentation_args_batchgenerators(self.parsed_args)


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
            raise ValueError("self.indices is None. Make sure to set it before calling this function!")

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
    """This uses the batchgenerators API and allows 3D data augmentation."""
    def __init__(self,
                 x,  # (n_samples, z, y, x, channels) for 3D or (n_samples, y, x, channels) for 2D data
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

        imgs = transformed["data"]  # is still in format (b, c, x, y, z) or 2d respectively
        axes = list(range(imgs.ndim))  # (0, 1, 2, 3) for 2D or (0, 1, 2, 3, 4) for 3D with batches and channels first
        transpose_axes = [axes[0]] + axes[1:][::-1]  # now (0, 3, 2, 1) or (0, 4, 3, 2, 1) which would be (b, z, y, x, c)
        imgs = np.transpose(imgs, axes=transpose_axes)

        if self.save_to_dir:
            os.makedirs(self.save_to_dir, exist_ok=True)
            for i, j in enumerate(index_array):
                img = imgs[i]
                fname = '{prefix}_index{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)

                plot_3d_array(
                    img, title=f"Sample {j}",
                    output_dir=os.path.join(self.save_to_dir, fname))

        if labels is None:
            return imgs
        else:
            return imgs, labels


class DataHandler3D(DataHandlerBase):
    def __init__(self,
                 input, outcome,
                 time_col,
                 event_col,
                 id_col="id",
                 max_time_perturb=0,
                 batch_size=32,
                 mask_nontumor_areas=False,
                 no_data_augmentation=True,
                 training_augmentation_args={},
                 validation_augmentation_args={},
                 ###
                 crop_size=(32, 64, 64),
                 n_random_crops_per_patient=0,
                 max_pixels_shift=(0, 0, 0)):
        """
        Parameters
        ----------
        crop_size: for (z, y, x) dimension
        """
        # slices around max will now be interpreted as the number
        # of random crops around the center
        self.n_random_crops = n_random_crops_per_patient
        self.crop_size = np.array(crop_size)
        self.max_pixels_shift = max_pixels_shift
        # needed for batchgenerators API: only the (z, y, x) dimension, no channels!
        self.patch_size = np.array(crop_size)

        super().__init__(
            input=input, outcome=outcome,
            time_col=time_col, event_col=event_col, id_col=id_col,
            max_time_perturb=max_time_perturb,
            batch_size=batch_size,
            mask_nontumor_areas=mask_nontumor_areas,
            no_data_augmentation=no_data_augmentation,
            training_augmentation_args=training_augmentation_args,
            validation_augmentation_args=validation_augmentation_args)

    def _read_data(self):
        # reads all CT and extracts all slices around tumor
        data = read_patient_data(
            img_directories=self.input,
            outcome_file=self.outcome_file,
            time_col=self.time_col,
            event_col=self.event_col,
            id_col=self.id_col,
            n_around_max=-1,
            preproc_fun=None)

        self.outcome_dict = data[0]
        self.img_dict = data[1]

        # check that we have outcome for all patients with images
        assert set(self.img_dict).issubset(set(self.outcome_dict))

        self.patient_ids = sorted(self.outcome_dict.keys())

        # now change the images to contain 3d crops
        if self.mask_nontumor_areas:
            print("Will mask all area outside tumor mask with zeros!")
        for pat in self.img_dict:
            img = self.img_dict[pat]["img"]
            mask = self.img_dict[pat]["mask"]

            # crop centers are now relative to the CT region that
            # contains all slices with tumor
            # so to convert z-coordinates to absolute values, we
            # need to check which z coordinate belongs to the relative
            # position of the crop center
            # print(pat, "\n=========")
            img_crops, mask_crops, crop_centers = crop_3d_around_tumor_center(
                img, mask, crop_size=self.crop_size,
                n_random=self.n_random_crops,
                max_pixels_shift=self.max_pixels_shift)

            selected_slices_z = self.img_dict[pat]["slice_idx"]  # absolute z-coordinates
            relative_z_crops = [crop_centers[i][0] for i in range(len(crop_centers))]
            # print("\tselected_slices_z", selected_slices_z)
            # print("\trelative_z_crops", relative_z_crops)
            crop_centers_absolute = [None] * len(crop_centers)
            for i in range(len(crop_centers)):
                relative_z = crop_centers[i][0]  # z-coordinate comes first
                # y and x location are unchanged
                y, x = crop_centers[i][1:]
                # NOTE: the relative_z coordinate can become negative or larger
                # than the selected number of slices for
                # the selected crop centers. That means, the center of a
                # crop will be an artificially introduced zero padded slice.
                # This means we can not really state which "original" slice
                # is the centre of the crop so we introduce nan
                if relative_z < 0 or relative_z >= len(selected_slices_z):
                    absolute_z = np.nan
                else:
                    absolute_z = selected_slices_z[relative_z]

                crop_centers_absolute[i] = [absolute_z, y, x]

            # store volume information and fraction of volume captured
            # by each crop
            full_volume = np.sum(mask)
            self.img_dict[pat]["full_tumor_volume"] = full_volume

            crop_volumes = np.array([None] * len(mask_crops))
            for i, m_c in enumerate(mask_crops):
                crop_volumes[i] = np.sum(m_c)
            crop_volume_fractions = crop_volumes / full_volume
            self.img_dict[pat]["crop_tumor_volume_fraction"] = crop_volume_fractions

            self.img_dict[pat]["img"] = img_crops
            self.img_dict[pat]["mask"] = mask_crops
            self.img_dict[pat]["crop_centers"] = crop_centers_absolute

            # correct slice idx entries
            # TODO: why do we do this?
            # we replace parenthesis to avoid confusion with lists
            # if we read back this info from file
            crop_ids = np.array(
                [pat + "_3d-cropcenter-" + str(c).replace("[", "(").replace("]", ")")
                 for c in crop_centers_absolute])
            # before overriding this information, we have to copy it
            # (don't just use '=' since this would use references)
            self.img_dict[pat]["selected_slices"] = [
                s for s in self.img_dict[pat]["slice_idx"]]
            self.img_dict[pat]["slice_idx"] = crop_ids

            if self.mask_nontumor_areas:
                print("Masking all nontumor areas!")
                self.img_dict[pat]["img"] *= mask_crops

    def _make_training_transforms(self):
        if self.no_data_augmentation:
            print("No data augmentation will be performed during training!")
            return []

        patch_size = self.patch_size[::-1]  # (x, y, z) order
        rot_angle_x = self.training_augmentation_args.get('angle_x', 15)
        rot_angle_y = self.training_augmentation_args.get('angle_y', 15)
        rot_angle_z = self.training_augmentation_args.get('angle_z', 15)
        p_per_sample = self.training_augmentation_args.get(
            'p_per_sample', 0.15)

        train_transforms = [
            SpatialTransform_2(
                patch_size, patch_size // 2,
                do_elastic_deform=self.training_augmentation_args.get(
                    'do_elastic_deform', True),
                deformation_scale=self.training_augmentation_args.get(
                    'deformation_scale', (0, 0.25)),
                do_rotation=self.training_augmentation_args.get(
                    'do_rotation', True),
                angle_x=(-rot_angle_x/360.*2*np.pi, rot_angle_x/360.*2*np.pi),
                angle_y=(-rot_angle_y/360.*2*np.pi, rot_angle_y/360.*2*np.pi),
                angle_z=(-rot_angle_z/360.*2*np.pi, rot_angle_z/360.*2*np.pi),
                do_scale=self.training_augmentation_args.get('do_scale', True),
                scale=self.training_augmentation_args.get('scale', (0.75, 1.25)),
                border_mode_data='nearest', border_cval_data=0,
                order_data=3,
                # border_mode_seg='nearest', border_cval_seg=0,
                # order_seg=0,
                random_crop=False,
                p_el_per_sample=self.training_augmentation_args.get(
                    'p_el_per_sample', 0.5),
                p_rot_per_sample=self.training_augmentation_args.get(
                    'p_rot_per_sample', 0.5),
                p_scale_per_sample=self.training_augmentation_args.get(
                    'p_scale_per_sample', 0.5))
        ]

        if self.training_augmentation_args.get("do_mirror", False):
            train_transforms.append(MirrorTransform(axes=(0, 1, 2)))

        train_transforms.append(
            BrightnessMultiplicativeTransform(
                self.training_augmentation_args.get('brightness_range', (0.7, 1.5)),
                per_channel=True, p_per_sample=p_per_sample))
        train_transforms.append(
            GaussianNoiseTransform(
                noise_variance=self.training_augmentation_args.get(
                    'gaussian_noise_variance', (0, 0.05)),
                p_per_sample=p_per_sample))
        train_transforms.append(
            GammaTransform(
                gamma_range=self.training_augmentation_args.get(
                    'gamma_range', (0.5, 2)),
                invert_image=False, per_channel=True,
                p_per_sample=p_per_sample))

        print("train_transforms\n", train_transforms)

        return train_transforms

    def create_generators(self, training_ids, validation_ids):

        train_imgs, _, train_labels, _ = self.stacked_np_array(training_ids)
        # print("in create_generators, len(training_ids)={}, len(validation_ids)={},"
        #       " train_imgs.shape={}, train_labels.shape={}"
        #       " valid_imgs.shape={}, valid_labels.shape={}".format(
        #         len(training_ids), len(validation_ids),
        #         train_imgs.shape, train_labels.shape,
        #         valid_imgs.shape, valid_labels.shape))

        train_gen = NumpyArrayIteratorUpTo3D(
            train_imgs, train_labels,
            transform=self._make_training_transforms(),
            batch_size=self.batch_size,
            shuffle=True)

        if len(validation_ids) > 0:
            valid_imgs, _, valid_labels, _ = self.stacked_np_array(validation_ids)
            # validation
            valid_gen = NumpyArrayIteratorUpTo3D(
                valid_imgs, valid_labels, transform=[],
                batch_size=self.batch_size,
                shuffle=False)
        else:
            valid_gen = None

        return train_gen, valid_gen


class FromScratchCVContext3D(FromScratchCVContext):
    def __init__(self,
                 data_handler,
                 seed=1,
                 train_ids=None,
                 train_fraction=None,
                 epochs=[10],
                 optimizer_cls=optimizers.Adam,
                 lr=[1.e-3],
                 loss=neg_cox_log_likelihood,
                 batchnorm=None,
                 lrelu=0.,
                 l1=0.,
                 l2=0.,
                 dropout=0.,
                 finalact="tanh",
                 ####
                 num_threads=5,
                 **kwargs):

        self.num_threads = num_threads

        super().__init__(
            data_handler,
            seed=seed,
            train_ids=train_ids,
            train_fraction=train_fraction,
            epochs=epochs,
            optimizer_cls=optimizer_cls,
            lr=lr,
            loss=loss,
            batchnorm=batchnorm,
            lrelu=lrelu,
            l1=l1,
            l2=l2,
            dropout=dropout,
            finalact=finalact,
            **kwargs)

    def create_compiled_model(self, input_shape):
        m = hosny_model_3d(
                input_shape=input_shape, n_outputs=1,
                final_act=self.finalact,
                bn=self.batchnorm,
                lrelu=self.lrelu,
                l1_reg=self.l1,
                l2_reg=self.l2)

        m.compile(
            loss=self.loss,
            optimizer=self.optimizer_cls(lr=self.lr[0]))

        return m

    def train_function(self, compiled_model, train_generator,
                       valid_generator, callbacks):
        early_stop = keras_cb.EarlyStopping(
            monitor="val_loss", patience=10, verbose=1, mode="min")
        reduce_lr = keras_cb.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, mode="min",
            min_lr=1e-7)
        cb_ci = ConcordanceIndex(
            (train_generator.x, train_generator.y),
            (valid_generator.x, valid_generator.y),
            freq=1)

        callbacks = [
            # early_stop, reduce_lr,
            cb_ci] + callbacks

        # make use of multiprocessing since 3d augmentation
        # takes quite a while
        hist = compiled_model.fit_generator(
                train_generator,
                steps_per_epoch=np.ceil(
                    train_generator.n / train_generator.batch_size),
                epochs=self.epochs[0],
                verbose=1,
                validation_data=valid_generator,
                validation_steps=np.ceil(
                    valid_generator.n / valid_generator.batch_size),
                callbacks=callbacks,
                use_multiprocessing=True,
                workers=self.num_threads)

        return hist

    def write_data_info(self, output_dir):

        # we need to write the crop centers to file
        patients = []
        centers = []
        volume_covered = []
        for pat in self.data_handler.img_dict:
            # center (z,y,x) for each crop of that patient
            cen = self.data_handler.img_dict[pat]["crop_centers"]
            # the fraction of tumor volume covered by each crop
            volume_fractions = self.data_handler.img_dict[pat][
                "crop_tumor_volume_fraction"]
            for i, coords in enumerate(cen):
                patients.append(pat)
                centers.append(coords)
                volume_covered.append(volume_fractions[i])

        df = pd.DataFrame({
            'id': patients,
            'crop_center': centers,
            'fraction_total_tumor_volume_covered': volume_covered
        })

        df.to_csv(os.path.join(output_dir, "crop_centers.csv"), index=False)


if __name__ == "__main__":
    run_cv_from_cmd(
        cv_context_cls=FromScratchCVContext3D,
        data_handler_cls=DataHandler3D,
        parser_cls=SurvivalCVCmdLineParserDataAug,
        ensemble_method=np.mean)
