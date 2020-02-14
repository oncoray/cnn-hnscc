import numpy as np

from cv_train_from_scratch_3d import DataHandler3D, SurvivalCVCmdLineParserDataAug
from cv_train_from_scratch import FromScratchCVContext

from dl_toolbox.cross_validation.run_cv_cmd import run_cv_from_cmd

from keras import optimizers


class DataHandler2D(DataHandler3D):
    """
    Uses same cropping and data augmentation as the 3d variant but collapses
    the 3d dimension into the sample dimension providing 2d samples instead
    of 3d samples.
    """
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
                 crop_size=(16, 224, 224),
                 n_random_crops_per_patient=0,
                 max_pixels_shift=(0, 0, 0)):

        super().__init__(
            input=input, outcome=outcome,
            time_col=time_col, event_col=event_col, id_col=id_col,
            max_time_perturb=max_time_perturb,
            batch_size=batch_size,
            mask_nontumor_areas=mask_nontumor_areas,
            no_data_augmentation=no_data_augmentation,
            training_augmentation_args=training_augmentation_args,
            validation_augmentation_args=validation_augmentation_args,
            crop_size=crop_size,
            n_random_crops_per_patient=0,
            max_pixels_shift=max_pixels_shift)

        # adapt the patch size since we want to work on 2D data only
        # by collapsing the z dimension into the samples
        self.patch_size = np.array(crop_size[1:])

    def _read_data(self):
        # now for each patient, we have a single 3d crop of data
        # and the shape is (1,) + crop_size + (n_channels,)
        super()._read_data()

        # get rid of the crop dimension in the front
        for pat in self.img_dict:
            self.img_dict[pat]["img"] = self.img_dict[pat]["img"][0]  # z, y, x, c
            self.img_dict[pat]["mask"] = self.img_dict[pat]["mask"][0]  # z, y, x, c
            self.img_dict[pat]["crop_centers"] = self.img_dict[pat]["crop_centers"][0]

            # fix the slice description through collapsing the 3d crop
            idx_crop = self.img_dict[pat]["slice_idx"][0]
            n_slices = self.img_dict[pat]["img"].shape[0]
            slice_idx = [idx_crop + "_slice_" + str(i)
                         for i in range(n_slices)]
            self.img_dict[pat]["slice_idx"] = np.array(slice_idx)


if __name__ == "__main__":
    run_cv_from_cmd(
        cv_context_cls=FromScratchCVContext,
        parser_cls=SurvivalCVCmdLineParserDataAug,
        data_handler_cls=DataHandler2D,
        ensemble_method=np.mean)
