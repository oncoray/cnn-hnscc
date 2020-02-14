import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from .read import read_patient_data
from ..preprocessing.preprocessing import stack_slices_for_patients


class DataHandlerBase(object):
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
                 validation_augmentation_args={}):

        self.input = input
        self.outcome_file = outcome
        self.time_col = time_col
        self.event_col = event_col
        self.id_col = id_col

        self.max_time_perturb = max_time_perturb

        self.batch_size = batch_size
        self.mask_nontumor_areas = mask_nontumor_areas
        self.no_data_augmentation = no_data_augmentation
        self.training_augmentation_args = training_augmentation_args
        self.validation_augmentation_args = validation_augmentation_args

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
        raise NotImplementedError

    def create_generators(self, training_ids, validation_ids):
        raise NotImplementedError

    def stacked_np_array(self, patient_ids=None, slice_idx_key="slice_idx"):
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


class DataHandler(DataHandlerBase):
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
                 slices_around_max=(0, 0)):

        self.n_around_max = slices_around_max
        super().__init__(
            input, outcome,
            time_col=time_col,
            event_col=event_col,
            id_col=id_col,
            max_time_perturb=max_time_perturb,
            batch_size=batch_size,
            mask_nontumor_areas=mask_nontumor_areas,
            no_data_augmentation=no_data_augmentation,
            training_augmentation_args=training_augmentation_args,
            validation_augmentation_args=validation_augmentation_args)

    def _read_data(self):
        """
        Parameters
        ----------
        base_dir: str
            path to the base directory that contains directories for cohorts
            that in turn contain patient
            folders with image data stored as numpy arrays
        cohorts: list of str
            the cohort directory names to read data from (have to be contained in base_dir)
        baseline_feat_filename: str or None
            name of a csv file within each cohort directory that contains patient features
            usable for some simple baseline prediction models
        """
        # READ INPUTS
        data = read_patient_data(
            img_directories=self.input,
            outcome_file=self.outcome_file,
            time_col=self.time_col,
            event_col=self.event_col,
            id_col=self.id_col,
            n_around_max=self.n_around_max,
            preproc_fun=None)

        self.outcome_dict = data[0]
        self.img_dict = data[1]

        # check that we have outcome for all patients with images
        assert set(self.img_dict).issubset(set(self.outcome_dict))

        self.patient_ids = sorted(self.img_dict.keys())

        if self.mask_nontumor_areas:
            print("Will mask all nontumor areas with zeros!")  # Backup of img is found under key 'img_bk'.")
            for pat in self.img_dict:
                mask = self.img_dict[pat]["mask"]
                # img = self.img_dict[pat]["img"]
                # self.img_dict[pat]["img_bk"] = img
                self.img_dict[pat]["img"] *= mask

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

    def create_generators(self, training_ids, validation_ids):

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
