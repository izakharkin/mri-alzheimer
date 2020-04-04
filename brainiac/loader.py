import pandas as pd
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import cv2
from scipy.ndimage.interpolation import rotate 

from .model_selection import ManualGroupKFold
from sklearn.preprocessing import LabelEncoder
    
def extract_groups(df, names):
    mask = df['Group'].map(lambda x: x in names)
    return df[mask].copy()

def encoder_it(df, mapper):
    df['Group'] = df['Group'].map(mapper)
    df['Subject'] = LabelEncoder().fit_transform(df['Subject'])
    return df

def train_test_split_df(df, classes, random_state=42):
    '''
        df: pd.DataFrame
        classes: list, e.x. ['CN', 'MCI', 'AD']
    '''
    mapper = {n:i for i, n in enumerate(classes)}
    df = encoder_it(extract_groups(df, classes), mapper)

    val_map = df['Group'].value_counts()
    inds = np.argsort(val_map.keys().to_list())
    levels = np.array(val_map.to_list())[inds]
    levels = [0.0]+[levels[:i+1].sum() for i in range(len(levels))]
    levels /= levels[-1]
    levels *= 100

    cv = ManualGroupKFold(n_splits=3, random_state=random_state)
    train_idx, test_idx = list(cv.split(df['Image Data ID'], df['Group'].values, df['Subject'].values))[0]
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    return df_train, df_test, levels


class ADNIClassificationDataset(data.Dataset):
    def __init__(
        self, 
        dataset_csv, 
        levels,
        method='train',
        target_size=(96, 96), 
        frames_to_take=128,
        images_path='/home/ADNI-processed/images/',
        train=False
#         start_frame=50,
#         end_frame=-50
    ):
        self.dataset_csv = dataset_csv  # pd.read_csv('/home/basimova_nf/ADNI-processed/data.csv')
        self.method = method
        self.target_size = np.asarray(target_size)
#         self.start_frame = start_frame
#         self.end_frame = end_frame
        self.frames_to_take = frames_to_take
        self.images_path = images_path
        self.train = train
        self.std = torch.tensor([1.])
        self.levels = levels


    def __len__(self):
        return len(self.dataset_csv)


    def __getitem__(self, index):
        row = self.dataset_csv.iloc[index]
        filename = row['File Names']
        im_path = f'{self.images_path}{filename}.npy'
        im = np.load(im_path, allow_pickle=True)
        
        if len(im.shape) == 4:
            im = im[0]
            
        im = np.transpose(im, (1, 0, 2))  # to get right projection axis ("up")
        diff = im.shape[0] - self.frames_to_take
        start = diff // 2
        end = -diff // 2 + (1 if diff % 2 != 0 else 0)
        im = im[start:end,:,:]

        new_im = np.zeros((im.shape[0], *self.target_size))
        for i, layer in enumerate(im):
            layer = cv2.resize(layer, (self.target_size[1], self.target_size[0]))
            new_im[i] = layer
        new_im = new_im / new_im.max()
            
#         if self.normalize:
#             im = im / im.max()
        
#         new_im = np.zeros((im.shape[0], *self.target_size))
#         for i, layer in enumerate(im):
#             layer = Image.fromarray(np.uint8(layer * 255))
#             layer = layer.resize((self.target_size[1], self.target_size[0]), Image.NEAREST) 
#             new_im[i] = np.array(layer)
        if self.train:
            new_im = rotate(new_im,np.random.randint(low=-90, high=90, size=1)[0] ,(1,2),reshape=False)
            new_im = rotate(new_im,np.random.randint(low=0, high=5, size=1)[0] ,(0,1),reshape=False)
            new_im = rotate(new_im,np.random.randint(low=-5, high=5, size=1)[0] ,(0,2),reshape=False)
            new_im += np.random.normal(0,0.05, new_im.shape)
            new_im = np.abs(np.flip(new_im, axis=np.random.randint(low=0, high=2, size=1)[0]))
        new_im = new_im[None,:]  # np.moveaxis(new_im, 0, -1)[None, :]
            
        if self.method == 'test': return new_im
        target = self.levels[row['Group']] + torch.abs(torch.normal(0, self.std))
        return torch.Tensor(new_im), target, row['Group'] #,f'{self.images_path}{filename}.npy', index
