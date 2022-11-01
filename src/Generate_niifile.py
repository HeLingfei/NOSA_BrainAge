#生成nii文件，结果可视化
import pandas as pd
import xlwt
import nibabel as nib
import numpy as np
from nilearn import plotting
atlas = nib.load('../analytical_data/BN_Atlas_246_combined_15mm.nii')
affine = atlas.affine
atlas = np.asarray(atlas.get_fdata(),dtype=np.float32)
result = np.zeros((atlas.shape))
y = pd.read_excel('C:/Users/CC/Desktop/cchen/IS/HCP_G1_IS.xls')
label = y.values
label = label[:,4]
label = label.tolist()
result = np.zeros((atlas.shape))
for i in range(1,247):
    temp = (atlas == i)
    a = temp.astype(int)
    result +=(a*label[i-1])
result = nib.Nifti1Image(result,affine)
nib.save(result,'C:/Users/CC/Desktop/cchen/IS/HCP_G1_IS.nii')



