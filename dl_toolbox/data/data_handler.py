import numpy as np

from warnings import warn
from keras.preprocessing.image import ImageDataGenerator

from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform,\
    BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform


from .read import read_patient_data
from ..preprocessing.preprocessing import stack_slices_for_patients, to_rgb,\
    normalize_img, crop_3d_around_tumor_center
from ..generators import batchgenerators


class DataHandlerBase(object):
    def __init__(self,
                 input, outcome,
                 time_col,
                 event_col,
                 id_col="id",
                 max_time_perturb=0,
                 batch_size=32,
                 mask_mode=None,
                 no_data_augmentation=True,
                 training_augmentation_args={},
                 validation_augmentation_args={},
                 **kwargs):

        self.input = input
        self.outcome_file = outcome
        self.time_col = time_col
        self.event_col = event_col
        self.id_col = id_col

        self.max_time_perturb = max_time_perturb

        self.batch_size = batch_size
        self.mask_mode = mask_mode
        self.no_data_augmentation = no_data_augmentation
        self.training_augmentation_args = training_augmentation_args
        self.validation_augmentation_args = validation_augmentation_args

        self.kwargs = kwargs

        # those have to be set when reading data
        self.patient_ids = None
        self.outcome_dict = None
        self.img_dict = None
        self.baseline_feat_df = None
        self.labels = None

        # flag for signaling whether data has been read or not
        self.data_ready = False

    def _make_label_dict_from_outcomes(self):
        # normally the labels correspond to the outcome dict
        # but in some cases it may be different
        # e.g. when using classification tasks
        return self.outcome_dict

    def _read_data(self):
        raise NotImplementedError

    def read_data(self):
        self._read_data()
        self.labels = self._make_label_dict_from_outcomes()
        self.data_ready = True

    def read_data_if_necessary(self, force_reread=False):
        # read data and store the patient ids
        # if it is not there or we are forced to reread
        if not self.data_ready or force_reread:
            print("DataHandler: will read data!")
            self.read_data()
        else:
            print("DataHandler: not going to re-read data!")

    def get_img_data_for_patient(self, patient_id, **kwargs):
        return self.img_dict[patient_id]

    def create_generators(self, training_ids, validation_ids, n_outputs=1):
        raise NotImplementedError

    def stacked_np_array(self, patient_ids=None, slice_idx_key="slice_idx"):
        """
        Returns
        -------
        A list containing the stacked slices of the ct as np.array, the stacked
        slices of the mask as np.array, the label of the patient now assigned to each
        slice and the id for each slice
        """
        if patient_ids is None:
            patient_ids = self.patient_ids

        # check that the ids are present in outcome and img data
        if not set(patient_ids).issubset(set(self.patient_ids)):
            raise ValueError(
                "No data available for requested patients {}".format(
                    set(patient_ids) - set(self.patient_ids)))

        return stack_slices_for_patients(
            self.img_dict, self.labels, patient_ids,
            max_time_perturb=self.max_time_perturb,
            slice_idx_key=slice_idx_key)

    def handle_mask_input(self):
        """
        Transforms the img stored in the 'img' key of the img_dict
        by either masking non-tumor areas or concatenating img and mask
        in the channel dimension
        """
        if self.mask_mode is None:
            return

        elif self.mask_mode == "tumor_only":
            print("Will mask all nontumor areas with zeros!")
            for pat in self.img_dict:
                mask = self.img_dict[pat]["mask"]
                # img = self.img_dict[pat]["img"]
                # self.img_dict[pat]["img_bk"] = img
                self.img_dict[pat]["img"] *= mask
        elif self.mask_mode == "channel":
            print("Will concatenate img and mask in the channel dimension!")
            for pat in self.img_dict:
                img = self.img_dict[pat]["img"]
                mask = self.img_dict[pat]["mask"]

                self.img_dict[pat]["img"] = np.concatenate(
                    [img, mask], axis=-1)
        else:
            raise ValueError(f"Invalid mask_mode {self.mask_mode}!")


class DataHandler(DataHandlerBase):
    """Handles generation of 2D samples with keras data augmentation."""

    def __init__(self,
                 input, outcome,
                 time_col,
                 event_col,
                 id_col="id",
                 max_time_perturb=0,
                 batch_size=32,
                 mask_mode=None,
                 no_data_augmentation=True,
                 training_augmentation_args={},
                 validation_augmentation_args={},
                 slices_around_max=(0, 0),
                 slices_as_channels=False,
                 **kwargs):

        self.n_around_max = slices_around_max
        self.slices_as_channels = slices_as_channels

        super().__init__(
            input, outcome,
            time_col=time_col,
            event_col=event_col,
            id_col=id_col,
            max_time_perturb=max_time_perturb,
            batch_size=batch_size,
            mask_mode=mask_mode,
            no_data_augmentation=no_data_augmentation,
            training_augmentation_args=training_augmentation_args,
            validation_augmentation_args=validation_augmentation_args,
            **kwargs)

    def _read_data(self):
        """
        Parameters
        ----------
        """
        # READ INPUTS
        data = read_patient_data(
            img_directories=self.input,
            outcome_file=self.outcome_file,
            time_col=self.time_col,
            event_col=self.event_col,
            id_col=self.id_col,
            n_around_max=self.n_around_max,
            preproc_fun=None,
            **self.kwargs)

        self.outcome_dict = data[0]
        self.img_dict = data[1]

        # check that we have outcome for all patients with images
        assert set(self.img_dict).issubset(set(self.outcome_dict))

        self.patient_ids = sorted(self.img_dict.keys())

        self.handle_mask_input()

        if self.slices_as_channels:
            # does not make sense if we concatenated img and mask
            # as input previously!
            if self.mask_mode == "channel":
                raise ValueError(
                    "'slices_as_channels' option not compatible with mask_mode 'channel'!")

            print("Will move slices to channel axis")
            for pat in self.img_dict:
                # n x h x w x 1
                img = self.img_dict[pat]["img"]
                mask = self.img_dict[pat]["mask"]

                # now 1 x h x w x n
                img_t = img.transpose((3, 1, 2, 0))
                mask_t = mask.transpose((3, 1, 2, 0))

                print(pat, img.shape, mask.shape,
                      "new:", img_t.shape, mask_t.shape)
                self.img_dict[pat]["img"] = img_t
                self.img_dict[pat]["mask"] = mask_t

    def get_img_data_for_patient(self, patient_id, slices=None):
        """
        Parameters
        ----------
        patient_id: str
            name of the patient
        slices: list of int
            the positions of the slices of the contained images
            (note: those should range from 0 to len(img_dict[patient_id]['img'])))
            If None, all slices of a patient will be chosen

        Returns
        -------
        a triple of numpy arrays containing the image slices, the mask slices and the
        indices that those slices had in the original full image (before being reduced to some
        slices around the tumor)
        """
        pat_dict = self.img_dict[patient_id]
        img = pat_dict["img"]
        mask = pat_dict["mask"]
        slice_idx = pat_dict["slice_idx"]

        if slices is not None:
            assert np.min(slices) >= 0
            assert np.max(slices) < len(img)

            img = img[slices]
            mask = mask[slices]
            slice_idx = slice_idx[slices]

        return img, mask, slice_idx

    def create_generators(self, training_ids, validation_ids, n_outputs=1):

        train_imgs, _, train_labels, _ = self.stacked_np_array(training_ids)
        valid_imgs, _, valid_labels, _ = self.stacked_np_array(validation_ids)

        if self.no_data_augmentation:
            print("No data augmentation will be done!")
            train_datagen = ImageDataGenerator()
            valid_datagen = ImageDataGenerator()
        else:
            train_datagen = ImageDataGenerator(
                **self.training_augmentation_args)
            valid_datagen = ImageDataGenerator(
                **self.validation_augmentation_args)

        train_generator = train_datagen.flow(
            train_imgs, train_labels,
            batch_size=self.batch_size, shuffle=True)

        valid_generator = valid_datagen.flow(
            valid_imgs, valid_labels,
            batch_size=self.batch_size, shuffle=False)

        return train_generator, valid_generator


class DataHandlerRGB(DataHandler):
    """A data handler to read rgb images instead of grayscale"""

    def __init__(self,
                 input, outcome,
                 time_col,
                 event_col,
                 id_col="id",
                 max_time_perturb=0,
                 batch_size=32,
                 mask_mode=None,
                 no_data_augmentation=True,
                 training_augmentation_args={},
                 validation_augmentation_args={},
                 slices_around_max=(0, 0),
                 preproc_fun=None,
                 **kwargs):

        self.preproc_fun = preproc_fun

        if mask_mode == "channel":
            raise ValueError(
                "mask_mode 'channel' incompatible with transfer learning, "
                "because RGB images are needed!")

        super().__init__(
            input=input, outcome=outcome,
            time_col=time_col, event_col=event_col, id_col=id_col,
            max_time_perturb=max_time_perturb,
            batch_size=batch_size,
            mask_mode=mask_mode,
            no_data_augmentation=no_data_augmentation,
            training_augmentation_args=training_augmentation_args,
            validation_augmentation_args=validation_augmentation_args,
            slices_around_max=slices_around_max,
            **kwargs)

    def _read_data(self):
        super()._read_data()

        def rgb_preproc(ct_slices, roi, label):
            # convert from [0,1] range to [0, 255]
            rgb_range = np.zeros(ct_slices.shape, dtype="uint8")
            for i, ct in enumerate(ct_slices):
                rgb_range[i] = normalize_img(
                    ct, new_low=0, new_up=255).astype("uint8")

            rgb = to_rgb(rgb_range)

            return rgb, roi, label

        # the preprocessing for the networks needs to be done!
        def network_preproc(ct_slices_rgb, roi, label):
            if self.preproc_fun is None:
                return ct_slices_rgb, roi, label

            ct_preproc = np.zeros(ct_slices_rgb.shape)
            for i, ct in enumerate(ct_slices_rgb):
                ct_preproc[i] = self.preproc_fun(ct)

            return ct_preproc, roi, label

        # now do the RGB conversion
        for pat in self.img_dict:
            # n_slices x height x width x 1
            img = self.img_dict[pat]['img']
            roi = self.img_dict[pat]['mask']
            lab = self.outcome_dict[pat]

            img_rgb, _, _ = rgb_preproc(img, roi, lab)
            img_pre, _, _ = network_preproc(img_rgb, roi, lab)

            # DEBUG:
            # f, ax = plt.subplots(2, 5, figsize=(15, 10))
            # slice_idx = len(img) // 2
            # ax[0, 0].imshow(img[slice_idx].squeeze(), cmap="gray")
            # ax[0, 0].set_title("original")

            # ax[0, 1].imshow(img_rgb[slice_idx])
            # ax[0, 1].set_title("after RGB")

            # ax[0, 2].imshow(img_rgb[slice_idx, ..., 0], cmap="gray")
            # ax[0, 2].set_title("channel 0 after RGB")

            # ax[0, 3].imshow(img_rgb[slice_idx, ..., 1], cmap="gray")
            # ax[0, 3].set_title("channel 1 after RGB")

            # ax[0, 4].imshow(img_rgb[slice_idx, ..., 2], cmap="gray")
            # ax[0, 4].set_title("channel 2 after RGB")

            # ax[1, 1].imshow(img_pre[slice_idx])
            # ax[1, 1].set_title("after RGB+preproc")
            # ax[1, 2].imshow(img_pre[slice_idx, ..., 0], cmap="gray")
            # ax[1, 2].set_title("channel 0 after RGB+preproc")
            # ax[1, 3].imshow(img_pre[slice_idx, ..., 1], cmap="gray")
            # ax[1, 3].set_title("channel 1 after RGB+preproc")
            # ax[1, 4].imshow(img_pre[slice_idx, ..., 2], cmap="gray")
            # ax[1, 4].set_title("channel 2 after RGB+preproc")
            # plt.show()
            ###########

            self.img_dict[pat]['img'] = img_pre
            # self.img_dict[pat]['mask'] = roi
            # self.outcome_dict[pat] = lab


class DataHandler3D(DataHandlerBase):
    """
    Works with precomputed 3D random crops before passing them to the
    generators for training.
    """

    def __init__(self,
                 input, outcome,
                 time_col,
                 event_col,
                 id_col="id",
                 max_time_perturb=0,
                 batch_size=32,
                 mask_mode=None,
                 no_data_augmentation=True,
                 training_augmentation_args={},
                 validation_augmentation_args={},
                 ###
                 crop_size=(32, 64, 64),
                 n_random_crops_per_patient=0,
                 max_pixels_shift=(0, 0, 0),
                 img_filenames=["ct"],
                 mask_filenames=["roi"],
                 **kwargs):
        """
        Parameters
        ----------
        crop_size: for (z, y, x) dimension
        img_filenames: list of str
                       Specifies which files to read within the <input> folder for each patient.
                       Each list entry should provide data for a different imaging time point.
                       The resulting self.img_dict will have as entries for a patient a list of
                       numpy files containing the image for each time point
        mask_filenames: list of str
                        Similar as above, except those are for the tumor segmentation masks.
        """
        # slices around max will now be interpreted as the number
        # of random crops around the center
        self.n_random_crops = n_random_crops_per_patient
        self.n_crops = self.n_random_crops + 1
        self.crop_size = np.array(crop_size)
        self.max_pixels_shift = max_pixels_shift
        # needed for batchgenerators API: only the (z, y, x) dimension, no channels!
        self.patch_size = np.array(crop_size)

        # only support single time point
        assert len(img_filenames) == len(mask_filenames)
        self.n_time_points = len(img_filenames)
        self.img_filenames = img_filenames
        self.mask_filenames = mask_filenames

        super().__init__(
            input=input, outcome=outcome,
            time_col=time_col, event_col=event_col, id_col=id_col,
            max_time_perturb=max_time_perturb,
            batch_size=batch_size,
            mask_mode=mask_mode,
            no_data_augmentation=no_data_augmentation,
            training_augmentation_args=training_augmentation_args,
            validation_augmentation_args=validation_augmentation_args,
            **kwargs)

    def _read_data(self):
        # reads all CT and extracts all slices with tumor
        data = read_patient_data(
            img_directories=self.input,
            outcome_file=self.outcome_file,
            time_col=self.time_col,
            event_col=self.event_col,
            id_col=self.id_col,
            n_around_max=-1,
            preproc_fun=None,
            img_fn=self.img_filenames[0],  # only support single time point
            mask_fn=self.mask_filenames[0],
            **self.kwargs)

        self.outcome_dict = data[0]
        self.img_dict = data[1]

        # check that we have outcome for all patients with images
        assert set(self.img_dict).issubset(set(self.outcome_dict))

        self.patient_ids = sorted(self.img_dict.keys())

        # now change the images to contain 3d crops
        volume_coverage = []
        for pat in self.img_dict:
            img = self.img_dict[pat]["img"]
            mask = self.img_dict[pat]["mask"]

            # crop centers are now relative to the CT region that
            # contains all slices with tumor
            # so to convert z-coordinates to absolute values, we
            # need to check which z coordinate belongs to the relative
            # position of the crop center
            print("\n", pat, "\n=========")
            img_crops, mask_crops, crop_centers = crop_3d_around_tumor_center(
                img, mask, crop_size=self.crop_size,
                n_random=self.n_random_crops,
                max_pixels_shift=self.max_pixels_shift)

            # absolute z-coordinates
            selected_slices_z = self.img_dict[pat]["slice_idx"]
            # print("slices with tumor", selected_slices_z)

            relative_z_crops = [crop_centers[i][0]
                                for i in range(len(crop_centers))]
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

            # print("\n", pat, "volume fractions\n", crop_volume_fractions)
            volume_coverage += list(crop_volume_fractions)

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

            # print("\n", pat, "crop_ids\n", crop_ids)

        print(
            f"\nVolume coverage over all {len(volume_coverage)} slices of all {len(self.img_dict)} patients\n")
        print(
            f"mean={np.mean(volume_coverage)}, median={np.median(volume_coverage)}")

        self.handle_mask_input()

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
                scale=self.training_augmentation_args.get(
                    'scale', (0.75, 1.25)),
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
                self.training_augmentation_args.get(
                    'brightness_range', (0.7, 1.5)),
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

    def create_generators(self, training_ids, validation_ids, n_outputs=1):

        if n_outputs > 1:
            warn("N_outputs > 1 not supported!")

        train_imgs, _, train_labels, _ = self.stacked_np_array(training_ids)
        # print("in create_generators, len(training_ids)={}, len(validation_ids)={},"
        #       " train_imgs.shape={}, train_labels.shape={}"
        #       " valid_imgs.shape={}, valid_labels.shape={}".format(
        #         len(training_ids), len(validation_ids),
        #         train_imgs.shape, train_labels.shape,
        #         valid_imgs.shape, valid_labels.shape))

        train_gen = batchgenerators.NumpyArrayIteratorUpTo3D(
            train_imgs, train_labels,
            transform=self._make_training_transforms(),
            batch_size=self.batch_size,
            shuffle=True)

        if len(validation_ids) > 0:
            valid_imgs, _, valid_labels, _ = self.stacked_np_array(
                validation_ids)
            # validation
            valid_gen = batchgenerators.NumpyArrayIteratorUpTo3D(
                valid_imgs, valid_labels, transform=[],
                batch_size=self.batch_size,
                shuffle=False)
        else:
            valid_gen = None

        return train_gen, valid_gen
