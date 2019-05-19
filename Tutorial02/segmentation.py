
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

from sklearn import mixture

from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.morphology import erosion, dilation, disk

import vtk

def segmentation_core():

    imag_directory = '../dcm'
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(imag_directory)
    if not series_IDs:
        print("ERROR: given directory \"" + data_directory + "\" does not contain a DICOM series.")
        sys.exit(1)
    
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(imag_directory, series_IDs[0])
    # print(series_file_names)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    images = series_reader.Execute()


    m_buffer = sitk.GetArrayFromImage(images)

    # thresholding
    mx = -300
    mn = -1100
    seg_buffer = (m_buffer - mn)/(mx-mn)
    seg_buffer[seg_buffer > 1.0] = 1.0
    seg_buffer[seg_buffer < 0.0] = 0.0

    # truncate ..
    seg_buffer[:70, :, :] = 0.0
    seg_buffer[295:, :, :] = 0.0

    # type conversion ..
    seg_buffer = seg_buffer.astype(np.float32)

    # gaussian mixture segmentation
    strIdx = 70
    endIdx = 295

    gm_buffer = np.zeros((seg_buffer.shape[1]*seg_buffer.shape[2], 2), dtype=seg_buffer.dtype)

    for idx in range(strIdx, endIdx):
        # initialize
        gm_buffer.fill(0.0)
        gm_buffer[:, 0] = seg_buffer[idx, :, :].flatten()

        gm_segment = np.zeros(shape=seg_buffer.shape[1]*seg_buffer.shape[2], dtype=seg_buffer.dtype).flatten()

        numIter = 1
        for iter in range(numIter):

            # fit !!
            dpgmm = mixture.GaussianMixture(n_components=2, covariance_type='full', max_iter=10).fit(gm_buffer)

            # prediction
            prob_map = dpgmm.predict_proba(gm_buffer)

            # convert to the distribution
            prob_map = np.transpose(np.divide(np.transpose(prob_map), np.sum(prob_map, axis=1)))
            prob_map[prob_map < 0.001] = 0.0

            if dpgmm.means_[0, 0] < dpgmm.means_[1, 0]:
                gm_segment += prob_map[:, 0]
            else:
                gm_segment += prob_map[:, 1]


        print("process finished at {:3d}".format(idx))

        gm_segment = np.divide(gm_segment.reshape(seg_buffer[idx, :, :].shape), numIter)
        gm_segment[gm_segment > 0.5] = 1.0
        gm_segment[gm_segment <= 0.5] = 0.0

        # small hole remove ..
        filled_segment = remove_small_holes(gm_segment.astype(np.bool), area_threshold=64*64)
        # small region remove ..
        filled_segment = remove_small_objects(filled_segment.astype(np.bool))

        # boundary mophology
        # selem = disk(4)
        filled_segment = dilation(filled_segment, selem=disk(4))
        filled_segment = erosion(filled_segment, selem=disk(3))

        # revert to the original shape
        seg_buffer[idx, :, :] = filled_segment.astype(np.float32)


    # additional process
    # filled_segment = remove_small_objects(filled_segment.astype(np.bool))
    initial_seed_point_indexes =[(144, 256, 130),(368, 256, 130)]
    seg_images = sitk.ConnectedThreshold(sitk.GetImageFromArray(seg_buffer), seedList=initial_seed_point_indexes, lower=0.5, upper=1.0)

    seg_buffer= sitk.GetArrayFromImage(seg_images)


    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    ax = axes.flatten()

    ax[0].imshow(seg_buffer[70, :, :], cmap="gray")
    ax[0].set_axis_off()

    ax[1].imshow(seg_buffer[120, :, :], cmap="gray")
    ax[1].set_axis_off()

    ax[2].imshow(seg_buffer[200, :, :], cmap="gray")
    ax[2].set_axis_off()

    ax[3].imshow(seg_buffer[294, :, :], cmap="gray")
    ax[3].set_axis_off()


    # image thresholding with shift
    seg_images = sitk.GetImageFromArray(np.multiply(m_buffer - np.amin(m_buffer), seg_buffer)+mn)
    seg_images.SetSpacing(images.GetSpacing())
    seg_images.SetOrigin(images.GetOrigin())
    sitk.WriteImage(seg_images, 'segmentation_final.mhd')

    fig.tight_layout()
    plt.show()


def rendering_core():
    filepath = 'segmentation_final.mhd'


if __name__=='__main__':
    segmentation_core()