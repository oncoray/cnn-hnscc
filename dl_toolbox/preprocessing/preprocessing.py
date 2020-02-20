import numpy as np
from PIL import Image
from skimage.util import pad
from skimage.measure import regionprops
from sklearn.feature_extraction import image


def to_rgb(imgs):
    """
    Replicate the last dimension 3 times.
    """
    # imgs should be N x H x W x 1
    # will output N x H x W x 3
    rgb = np.zeros((imgs.shape[:3]) + (3,), dtype=imgs.dtype)

    for i, img in enumerate(imgs):
        rep = np.repeat(img, 3, axis=2)
        #print(rep.shape)
        assert np.all(rep[:, :, 0] == rep[:, :, 1])
        assert np.all(rep[:, :, 1] == rep[:, :, 2])
        rgb[i] = rep

    return rgb


def resize_img(img_array, target_shape):
    # img should be 2d numpy array representing an image of H X W
    # print("resize_img: got shape", img_array.shape, "to transform to", target_shape)
    img = Image.fromarray(img_array)
    img = img.resize(target_shape)

    return np.array(img)


def clip_range(img, hu_range):
    lower, upper = hu_range

    img[img < lower] = lower
    img[img > upper] = upper

    return img


def standardize_ct(img):
    # img should be 2d np array representing an image

    # standardize
    m = np.mean(img)
    sd = np.std(img)

    if sd != 0:
        res = (img - m) / sd
    else:
        # sd = 0 => img is constant => m = img => res = 0
        res = img - m

    return res


def normalize_img(img, new_low=0, new_up=1):
    # img should be 2d np.array or 3d with single channel
    m = img.min()
    M = img.max()

    img_normalized = (img - m) / (M - m) * (new_up - new_low) + new_low
    return img_normalized


def pandas_outcome_to_dict(df, id_col='id_subject',
                           survival_col='locoRegionFreeSurv',
                           event_col='locoRegionRecurrence',
                           ids=None):  # the ids to find outcome for
    outcomes = {}
    colnames = [survival_col, event_col]  # names of columns in outcome dataframe for survival_time and event_status

    if ids is None:
        ids = df[id_col].values

    for pat_id in ids:
        # match for our id
        res = df[df[id_col] == pat_id]
        if res.empty:
            print("No outcome for patient {0}! Will skip it.".format(pat_id))
            continue
        # take only the columns that are specified
        res = res.loc[:, colnames]
        t = float(res[survival_col])
        e = int(res[event_col])

        outcomes[pat_id] = (t, e)

    return outcomes


def extract_slices_around_largest_tumor(img, roi, n_around_max=23):
    """
    Parameters
    ----------
    img: np.array (3 dimensional, HxWxL or 4 dimensional HxWxLx1)
    roi: np.array (of same shape as img)
    n_around_max: int or tuple of length 2 of ints
        the number of slices before and after the slice with largest
        tumor area.
        if a tuple, the first number is used before the slice with
        largest tumor, the second one after.
        If a negative integer, all slices with tumor are extracted

    Returns
    -------
    a tuple of length three which contains maximally the 2*(n_around_max+1)
    image slices and the 2*(n_around_max+1) corresponding mask slices and the slice indices
    that correspond to the extracted slices.
    If not enough slices can be accessed, the amount is limited to the availble range.
    """
    # determine row with largest tumor area
    areas_per_row = np.sum(roi, axis=(1, 2, 3))
    idx_max = np.argmax(areas_per_row)

    if not isinstance(n_around_max, tuple):
        # convert single number to tuple
        n_around_max = (n_around_max, n_around_max)

    if len(n_around_max) != 2:
        raise ValueError(
            "n_around_max has to be a tuple of length 2 or a single int!")

    if n_around_max[0] < 0 or n_around_max[1] < 0:
        # extract all tumor slices
        tumor_rows = np.where(areas_per_row > 0)[0]
        return (img[tumor_rows, :, :, :],
                roi[tumor_rows, :, :, :],
                tumor_rows)

    else:
        start_row = idx_max - n_around_max[0]
        end_row = idx_max + n_around_max[1]
        # limit to boundaries given
        if start_row < 0:
            start_row = 0
            print("WARNING: limited lower bound!")
        if end_row > img.shape[0] - 1:
            end_row = img.shape[0] - 1
            print("WARNING: limited upper bound!")

        # print("row_max = ", idx_max, ", selecting slices from",
        #      start_row, "to", end_row)
        extracted_idx = np.array(range(start_row, end_row+1))

        return (img[start_row:(end_row+1), :, :, :],
                roi[start_row:(end_row+1), :, :, :],
                extracted_idx)


def stack_slices_for_patients(data_dict, patient_label_dict, ids=None,
                              max_time_perturb=0,
                              img_key="img", mask_key="mask",
                              slice_idx_key=None,
                              additional_feature_df=None,
                              id_col="id"):
    """
    Stacks all img slices below each other for all patients of the data_dict
    given by ids

    Parameters
    ----------
    data_dict: dict
        mapping of patient ids to a dict containing
        the np.array that contains the ct slices and
        the np.array that contains the mask slices.
        Keys can be specified via img_key and mask_key arguments.
    patient_label_dict: dict
        mapping patient ids to outcome (survival, event)
    ids: list
        list of ids for which the slices should be concatenated.
        If None, then all keys in data_dict will be used
    max_time_perturb: float
        The upper range of values that the survival times of a single
        patient are adjusted.
        A random value in the interval [-max_time_perturb, max_time_perturb]
        is chosen and added to each slice of the patients survival data.
        This can be useful to prevent ties in event times which have
        to be taken into account when computing cox partial likelihood.
        This assumes the survival times to be in the first column of the label.
    img_key: str, default: 'img'
        dictionary key where images are found from within data_dict patient entries
    mask_key: str, default: 'mask'
        dictionary key where segmentation masks are found from within data_dict patient entries
    slice_idx_key: str or None
        dictionary key where slice identifier strings are found from within data_dict patient entries.
        If None, slices will be enumerated
    additional_feature_df: pd.DataFrame or None
        a dataframe with additional (clinical) features per patient
    id_col: str, default 'id'
        name of the column in additional_feature_df containing the patient ids

    Returns
    -------
    A list containing the stacked slices of the ct as np.array, the stacked
    slices of the mask as np.array, the label of the patient now assigned to each
    slice and the id for each slice. In case additional_feature_df is not None,
    the stacked features of the patient are also returned at the last position.
    The stacking is peformed on the alphabetically sorted ids.
    """
    img_stack = []
    roi_stack = []
    labels_stack = []
    ids_stack = []

    if additional_feature_df is not None:
        feature_stack = []

    if ids is None:
        ids = list(data_dict.keys())

    ids = sorted(ids)

    for pat in ids:
        cts = data_dict[pat][img_key]
        rois = data_dict[pat][mask_key]

        n_slices = cts.shape[0]

        pat_label = patient_label_dict[pat]
        # replicate the label for each slice and the id
        # but handle the case where the label is a single number only
        if np.array(pat_label).ndim == 0:
            # a single value
            label_format = (n_slices,)
        else:
            label_format = (n_slices, len(pat_label))
        label = pat_label * np.ones(label_format)
        # in case we now have a single dimensional label, we expand it so the vstack later
        # on works as expected
        if label.ndim == 1:
            label = np.expand_dims(label, -1)

        # randomly perturb survival times to avoid exactly same label
        if max_time_perturb > 0:
            time_perturb = np.random.uniform(
                -max_time_perturb, max_time_perturb, n_slices)
            label[:, 0] += time_perturb

        if slice_idx_key is not None:
            slice_idx = [pat + "_slice_" + str(id) for id in data_dict[pat][
                slice_idx_key]]
        else:
            slice_idx = np.array([pat + "_slice_" + str(i) for i in range(
                n_slices)])

        img_stack.append(cts)
        roi_stack.append(rois)
        labels_stack.append(label)
        ids_stack.append(slice_idx)

        if pat not in patient_label_dict:
            raise ValueError(pat, "in data_dict but not in label_dict!")

        additional_features = None
        if additional_feature_df is not None:
            if pat not in additional_feature_df[id_col].values:
                raise ValueError(
                    "{} has no entry in additional_feature_df!".format(pat))

            additional_features = additional_feature_df[
                additional_feature_df[id_col] == pat].drop(id_col, axis=1).values[0]

            # replicate the additional features for each slice
            additional_features = np.tile(additional_features, (n_slices, 1))

            feature_stack.append(additional_features)

    ret_val = [np.vstack(img_stack), np.vstack(roi_stack),
               np.vstack(labels_stack), np.hstack(ids_stack)]

    if additional_feature_df is not None:
        ret_val.append(np.vstack(feature_stack))

    return ret_val


def convert_to_grid_img(img_stack, pad=0):
    """
    Takes a batch of images and creates a larger image by creating
    a quadratic grid of the individual images

    Parameters
    ----------
    img_stack: np.array of shape n_images x height x width x channels
    pad: number of pixels with zero value between images (in height and width)
    """
    n_slices, n_rows, n_cols, n_channels = img_stack.shape

    grid_size = int(np.ceil(np.sqrt(n_slices)))
    #print("grid_size", grid_size, "\n")
    grid_shape = (grid_size * n_rows + (grid_size+1)*pad, grid_size * n_cols + (grid_size+1)*pad, n_channels)

    # if we dont have enough slices to fill the grid we have to pad with zeros
    grid_elems_to_pad = int(grid_size ** 2 - n_slices)
#    if grid_elems_to_pad > 0:
#        print("have to pad {} grid elements with zeros for a {} x {} grid".format(
#            grid_elems_to_pad, grid_size, grid_size))

    grid = np.zeros(grid_shape)

    # fit the single images into the grid, the rest remain zeros
    for i, img in enumerate(img_stack):
        grid_row = i // grid_size
        grid_col = i % grid_size
        # print("img", i, "goes to row", grid_row, "col", grid_col)
        row_start = grid_row * n_rows + (grid_row + 1) * pad
        row_end = row_start + n_rows

        col_start = grid_col * n_cols + (grid_col + 1) * pad
        col_end = col_start + n_cols
        # print("row_start", row_start, "row_end", row_end, "col_start", col_start, "col_end", col_end)
        # print("_______________________")
        grid[row_start: row_end, col_start:col_end, :] = img

    return grid


def stack_3d_patches_for_patients(data_dict, patient_label_dict, patch_shape,
                                  stride, ids=None, remove_zero_patches=True,
                                  max_time_perturb=0):
    """
    Converts each 3d array of a patient into a number of smaller 3d patches.
    Then stacks all 3d patches below each other for all patients of the
    data_dict given by ids.

    Parameters
    ----------
    data_dict: dict
        mapping of patient ids to a dict containing
        the np.array that contains the ct slices under the 'ct' key and
        the np.array that contains the mask slices under the 'roi' key

    patient_label_dict: dict
        mapping patient ids to outcome (survival, event)

    patch_shape: tuple of length 3
        the shape of each

    stride: int
        the offset between two consecutive patches (applied along every dimension)

    ids: list
        list of ids for which the slices should be concatenated.
        If None, then all keys in data_dict will be used

    remove_zero_patches: bool
        flag to discard the patches that contain only zeros

    max_time_perturb: float
        The upper range of values that the survival times of a single
        patient are adjusted.
        A random value in the interval [-max_time_perturb, max_time_perturb]
        is chosen and added to each slice of the patients survival data.
        This can be useful to prevent ties in event times which have
        to be taken into account when computing cox partial likelihood.
        This assumes the survival times to be in the first column of the label.


    Returns
    -------
    A tuple containing the stacked 3d patches of the ct as np.array,
    the label of the patient now assigned to each patch and the id for each patch.
    """
    img_stack = []
    roi_stack = []
    labels_stack = []
    ids_stack = []

    if ids is None:
        ids = list(data_dict.keys())

    for pat in ids:
        if pat not in patient_label_dict:
            raise ValueError(pat, "in data_dict but not in label_dict!")
        pat_label = patient_label_dict[pat]

        # is 4d array H x W x L x 1
        cts = data_dict[pat]['ct']
        # a 6d array with first three dims specifying the position of the
        # patch in the grid
        patches_3d = create_3d_patches(cts.squeeze(), patch_shape, stride)
        # collapse the first 3 dimensions into 1
        patches_3d = patches_3d.reshape(
            (np.prod(patches_3d.shape[:3]),) + patches_3d.shape[3:])

        if remove_zero_patches:
            nonzero_idx = []
            n = patches_3d.shape[0]
            for i, patch in enumerate(patches_3d):
                if not np.all(patch == 0.):
                    nonzero_idx.append(i)
            print("{}: discarded {} slices with only zeros: {}".format(
                pat, n - len(nonzero_idx),
                set(range(n)) - set(nonzero_idx)))
            patches_3d = patches_3d[nonzero_idx]

        n_patches = patches_3d.shape[0]
        # replicate the label for each patch and the id
        label = pat_label * np.ones((n_patches, len(pat_label)))
        # randomly perturb survival times to avoid exactly same label
        if max_time_perturb > 0:
            time_perturb = np.random.uniform(
                -max_time_perturb, max_time_perturb, n_patches)
            label[:, 0] += time_perturb

        id_tmp = np.array([pat] * n_patches)

        img_stack.append(patches_3d)
        labels_stack.append(label)
        ids_stack.append(id_tmp)

    return (np.expand_dims(np.vstack(img_stack), -1),
            np.vstack(labels_stack),
            np.hstack(ids_stack))


def stack_3d_patches_around_tumor(data_dict, patient_label_dict, patch_shape,
                               stride, ids=None, max_time_perturb=0):
    """
    Converts each 3d array of a patient into a number of smaller 3d patches,
    which are all taken in the vicinity of the tumor.
    Then stacks all 3d patches below each other for all patients of the
    data_dict given by ids.

    Parameters
    ----------
    data_dict: dict
        mapping of patient ids to a dict containing
        the np.array that contains the ct slices under the 'ct' key and
        the np.array that contains the mask slices under the 'roi' key

    patient_label_dict: dict
        mapping patient ids to outcome (survival, event)

    patch_shape: tuple of length 3
        the shape of each

    stride: int
        the offset between two consecutive patches (applied along every dimension)

    ids: list
        list of ids for which the slices should be concatenated.
        If None, then all keys in data_dict will be used

    max_time_perturb: float
        The upper range of values that the survival times of a single
        patient are adjusted.
        A random value in the interval [-max_time_perturb, max_time_perturb]
        is chosen and added to each slice of the patients survival data.
        This can be useful to prevent ties in event times which have
        to be taken into account when computing cox partial likelihood.
        This assumes the survival times to be in the first column of the label.


    Returns
    -------
    A tuple containing the stacked 3d patches of the ct area close to the tumor as np.array,
    the label of the patient now assigned to each patch and the id for each patch.
    """

    def adjust_roi_range(lower, upper, min_size, lower_limit, upper_limit):
        """
        Parameters
        ----------
        lower: int
            the array index describing the start of the roi
            along given dimension
        upper: int
            the array index describing one above the end of the roi
            along given dimension
        min_size: int
            minimal number of slices between lower and upper that need
            to be used (and adjusting lower and upper bounds if needed)
        lower_limit: int
            the smallest number the lower parameter is allowed to take
            after adjustment (usually 0)
        upper_limit: int
            the largest number the upper parameter is allowed to take
            after adjustment
        """
        current_size = upper - lower

        if current_size < min_size:
            if upper_limit - lower_limit < min_size:
                raise ValueError(
                    "No chance to increase interval {} to size {}"
                    " within the limits {}".format(
                        (lower, upper), min_size, (lower_limit, upper_limit)))

            # enlarge both interval ends evenly as long as possible
            size = current_size
            i = 0
            while lower > lower_limit and upper < upper_limit and size < min_size:
                if i % 2:
                    lower -= 1
                else:
                    upper += 1
                i += 1
                size = upper - lower

            # check which one is at its max
            if size < min_size:
                # at least one needs to be at its limit but not both since
                # there would have been an exception
                diff = min_size - size

                if lower <= lower_limit:
                    # need to increase upper
                    upper += diff

                elif upper >= upper_limit:
                    # need to decrease lower
                    lower -= diff

        return lower, upper

    img_stack = []
    roi_stack = []
    labels_stack = []
    ids_stack = []

    if ids is None:
        ids = list(data_dict.keys())

    for pat in ids:
        if pat not in patient_label_dict:
            raise ValueError(pat, "in data_dict but not in label_dict!")
        pat_label = patient_label_dict[pat]

        # is 4d array H x W x L x 1
        cts = data_dict[pat]['ct']
        roi = data_dict[pat]['roi']

        # get the indices that limit tumor extension in all three dimensions
        # the upper bound is already one above the last index with tumor
        properties = regionprops(roi)[0]
        z_min, y_min, x_min, z_max, y_max, x_max = properties.bbox
        # rounded to valid indices
        #center_of_mass = np.round(properties.centroid)
        # all tumor coordinates (n_pixels x 3) for z, y, x coordinate
        #tumor_coords = properties.coords
        print("\n", pat, ": dimension limits of roi", (z_min, z_max), (y_min, y_max), (x_min, x_max))

        # if some dimension now is smaller than the patch shape we still have to use the patch shape
        z_min, z_max = adjust_roi_range(
            z_min, z_max, patch_shape[0], lower_limit=0, upper_limit=roi.shape[0])
        y_min, y_max = adjust_roi_range(
            y_min, y_max, patch_shape[1], lower_limit=0, upper_limit=roi.shape[1])
        x_min, x_max = adjust_roi_range(
            x_min, x_max, patch_shape[2], lower_limit=0, upper_limit=roi.shape[2])

        print("adjusted limits to patch size {}:".format(
            patch_shape), (z_min, z_max), (y_min, y_max), (x_min, x_max))

        # now we limit the image to the relevant area where tumor is visible
        cts_crop = cts[z_min:z_max, y_min:y_max, x_min:x_max, ]
        print("limited the ct to", cts_crop.shape)

        # a 6d array with first three dims specifying the position of the
        # patch in the grid
        patches_3d = create_3d_patches(cts_crop.squeeze(), patch_shape, stride)
        # collapse the first 3 dimensions into 1
        patches_3d = patches_3d.reshape(
            (np.prod(patches_3d.shape[:3]),) + patches_3d.shape[3:])
        print("extracted patches of shape {}".format(patches_3d.shape))

        n_patches = patches_3d.shape[0]
        # replicate the label for each patch and the id
        label = pat_label * np.ones((n_patches, len(pat_label)))
        # randomly perturb survival times to avoid exactly same label
        if max_time_perturb > 0:
            time_perturb = np.random.uniform(
                -max_time_perturb, max_time_perturb, n_patches)
            label[:, 0] += time_perturb

        id_tmp = np.array([pat] * n_patches)

        img_stack.append(patches_3d)
        labels_stack.append(label)
        ids_stack.append(id_tmp)

    return (np.expand_dims(np.vstack(img_stack), -1),
            np.vstack(labels_stack),
            np.hstack(ids_stack))




def create_3d_patches(arr_3d, chunk_shape, stride):
    """
    From a larger 3d array extract a number of smaller 3d patches with a given stride
    (and potentially zero pad beforehand)

    Parameters
    ----------
    arr_3d: np.array of shape H x W x L
    chunk_shape = np.array of ints with shape Hsmall x Wsmall x Lsmall (each entry not larger than original size)
    stride: int
    """
    def n_output_patches():
        """
        the number of patches in (z,y,x) dimension obtained when moving through an input array of shape 'ori_shape'
        with a stride of 'stride' and a target shape of 'chunk_shape'.

        This can give non-integer values if stride, chunk_shape and original shape do not fit to each other.

        The formula for the number of output patches along an arbitrary dimension is as follows:
        out[i] = 1+ (ori_shape[i] - chunk_shape[i]) / stride

        """
        return 1 + (np.array(ori_shape) - np.array(chunk_shape)) / stride

    def necessary_padding():
        """
        The amount of padding (pre and post) necessary for each dimension to allow the given stride and chunk_shape with
        the given arr_3d.shape.

        The number of output patches is rounded towards the next highest integer. Then we compute the new total length that
        would be required to yield this number k of patches of shape 'chunk_shape' with the given stride.

        We want that the last chunk (k) ends at the last element of the (padded) array:
        the total length is (ori_shape[i] + pad_total) where i is the dimension index (0, 1 or 2).
        the last chunk starts at position stride * (k-1) and ends at stride * (k-1) + chunk_shape[i] - 1.
        So we have to solve
            ori_shape[i] + pad_total - 1 = stride * (k-1) + chunk_shape[i] - 1
        <=> pad_total = chunk_shape[i] - ori_shape[i] + (k-1) * stride

        """
        output_shape = n_output_patches()

        padding = [None] * len(output_shape)
        for i, s in enumerate(output_shape):
            if s.is_integer():
                padding[i] = (0, 0)
            else:
                # round the output shape to the next highest integer
                # and determine the padding that leads to that shape
                total_pad = int(
                    stride * (np.ceil(s) - 1) + (chunk_shape[i] - ori_shape[i]))

                # determine pre/post padding
                if total_pad % 2 == 0:
                    # we have an even number of zeros to pad
                    # and can distribute to pre and post
                    padding[i] = (int(total_pad/2), int(total_pad/2))
                else:
                    # uneven number, we only apply padding previous to image
                    padding[i] = (total_pad, 0)

        return padding

    ori_shape = arr_3d.shape
    # list of tuples of length 2 (for each dimension that can be padded)
    paddings = necessary_padding()
    print("will apply paddings", paddings)

    # apply padding on the dimensions with a non-zero tuple
    arr_3d_pad = pad(arr_3d, paddings, mode='constant', constant_values=0)

    return image.extract_patches(
        arr_3d_pad, chunk_shape, extraction_step=stride)


def crop_3d(img, crop_size, center):
    """
    Crop a sample of shape crop_size from img with the central point of the crop
    located at 'center' coordinates.
    If the center + crop_size exceeds some of the image dimensions, zero padding
    is performed.

    Parameters
    ----------
    img: np.array of shape N x H x W x C or (z, y, x, c)
    crop_size: tuple/array of length 3 for specifying the crop shape along (z, y, x)
    """
    crop_size = np.array(crop_size).astype(int)
    center = np.array(center).astype(int)

    assert len(center) == len(crop_size) == 3

    half_size = crop_size // 2
    lower = center - half_size
    upper = lower + crop_size

    # check that the lower indices are still >= 0
    # and that the upper indices are all < img.shape

    padding_lower = -1 * np.minimum(0, lower)
    padding_upper = np.maximum(0, upper - img.shape[:-1])  # do not use the channels
    # no padding if img.shape >= upper <=> upper - img.shape <= 0
    # and leave positive values as they are if upper > img.shape <=> upper - img.shape > 0

    # assign no paddings to the last dimension which are channels
    padding_lower = list(padding_lower)
    padding_upper = list(padding_upper)
    padding_lower.append(0)
    padding_upper.append(0)

    #print("crop size desired = {}, center={} and img.shape={} requires padding lower={}, upper={}".format(
    #    crop_size, center, img.shape, padding_lower, padding_upper))
    #print("lower={}, upper={}".format(lower, upper))
    # have to create a list of length 3 containing tuples of length two with lower and upper value
    # for each dimension
    img_pad = np.pad(img, list(zip(padding_lower, padding_upper)),
                     mode='constant', constant_values=0.)
    #print("Padded image has shape {}".format(img_pad.shape))

    # now we still have to shift lower and upper by the padding used
    # to compensate for the changed image dimensions due to padding
    # and this shift is achieved by adding padding_lower to all coordinates
    lower = lower + np.array(padding_lower[:-1])
    upper = upper + np.array(padding_lower[:-1])#np.array(padding_upper[:-1])
    #print("borders for padded image: lower={}, upper={}".format(lower, upper))
    return img_pad[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]




def crop_3d_around_tumor_center(ct, mask, crop_size, n_random=0, max_pixels_shift=(0, 0, 0)):
    """
    Creates a 3D crop of shape <crop_size> with the tumors center
    of mass as the central point.
    If n_random > 0, additional crop centers will be selected randomly
    from the bounding box around the tumor center of mass which has radius of max_pixels_shift
    (along each dimension)
    """
    if len(max_pixels_shift) == 1:
        max_pixels_shift = max_pixels_shift * 3
    elif isinstance(max_pixels_shift, int):
        max_pixels_shift = [max_pixels_shift] * 3

    max_pixels_shift = np.array(max_pixels_shift)
    assert len(max_pixels_shift) == 3
    assert np.all(max_pixels_shift >= 0)

    assert np.all(np.array(crop_size) > 0)

    if n_random < 0:
        n_random = 0

    full_volume = np.sum(mask)
    print(f"crop_3d_around_tumor_center: ct.shape={ct.shape}, full_volume: {full_volume}")
    # locate the center of mass of the tumor
    properties = regionprops(mask.squeeze())[0]
    center_of_mass = np.round(properties.centroid).astype(int)
    central_crop = crop_3d(ct, crop_size, center_of_mass)
    central_mask = crop_3d(mask, crop_size, center_of_mass)

    print(f"crop_3d_around_tumor_center: central crop: center: {center_of_mass} "
          f"volume: {np.sum(central_mask)}, volume_fraction: {np.sum(central_mask)/full_volume}")

    all_crops = [central_crop]
    all_masks = [central_mask]

    crop_centers = [center_of_mass]
    for i in range(n_random):
        # randomly select offset vector
        offset = np.random.uniform(
            low=-max_pixels_shift, high=max_pixels_shift).astype(int)

        center = center_of_mass + offset
        crop = crop_3d(ct, crop_size, center)
        crop_mask = crop_3d(mask, crop_size, center)

        crop_volume = np.sum(crop_mask)
        print(f"crop_3d_around_tumor_center: random crop {i+1}: "
              f"center: {center}, volume: {crop_volume} "
              f"volume fraction: {crop_volume/full_volume}")

        all_crops.append(crop)
        all_masks.append(crop_mask)
        crop_centers.append(center)

    return np.array(all_crops), np.array(all_masks), np.array(crop_centers)
