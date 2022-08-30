import SimpleITK as sitk
import numpy as np
def get_spinedata(dataDir, pedicle_points,pedicle_points_in_zyx):
    """data_prepare

    Args:
        dataDir (str): nii路径
        pedicle_points (np): 椎弓根中心点坐标
        pedicle_points_in_zyx (bool): 坐标是否是zyx形式

    Returns:
        _type_: [掩模numpy, 各体素的坐标, 椎弓根中心点], 体素间距
    """
    mask = sitk.ReadImage(dataDir)
    spacing = list(mask.GetSpacing())
    tmp = spacing[0]
    spacing[0] = spacing[2]
    spacing[2] = tmp
    if pedicle_points_in_zyx:
        temp = pedicle_points[:, 0].copy()
        pedicle_points[:, 0] = pedicle_points[:, 2]
        pedicle_points[:, 2] = temp

    mask_array = sitk.GetArrayFromImage(mask)
    mask_array = np.transpose(mask_array, (2, 1, 0))
    mask_coards = np.zeros((3,mask_array.shape[0],mask_array.shape[1],mask_array.shape[2]))
    for i in range(mask_coards[0].shape[0]):
        mask_coards[0][i,:,:] = i
    for i in range(mask_coards[1].shape[1]):
        mask_coards[1][:,i,:] = i 
    for i in range(mask_coards[2].shape[2]):
        mask_coards[2][:,:,i] = i
    return {"mask_coards":mask_coards, 
            "mask_array":mask_array, 
            "pedicle_points":pedicle_points}, spacing