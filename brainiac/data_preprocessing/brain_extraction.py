#/usr/bin/python

import dicom2nifti
import nibabel as nib
from nipype.interfaces import fsl
import numpy as np
import os
from pathlib import Path


path_searh = '/home/ADNI'
save_path = '/home/ilja8800/ADNI_cut/'
home_path = os.path.abspath('.')

all_dirs = set()

for path in Path(path_searh).rglob('*.dcm'):
    all_dirs.add(str(path.parent))

for dir_name in all_dirs:
    name = dir_name.replace('/', '_')
    os.mkdir('new/'+name)
    dicom2nifti.convert_directory(dir_name, 'new/'+name)

os.chdir('../new')
for dir_name in os.listdir('.'):
    try:
        file_name = os.listdir(dir_name)[0]
        source_file = dir_name + '/' + file_name
        end_file = dir_name + '/' + file_name.replace('.nii.gz', '_bet')
        mybet = fsl.BET(in_file=source_file, out_file=end_file)
        result = mybet.run()
        img = np.array(nib.load(end_file+'.nii.gz').get_fdata())
        res_path = save_path + dir_name
        np.save(res_path, img)
    except:
        pass

os.chdir(home_path)
