## 3D Convolutional Neural Networks for MRI Brain Classification
![](./pics/brain.pdf)

The code was written by [Natasha Basimova](https://github.com/pigunther), [Nikita Mokrov](https://github.com/Tismoney), [Ilya Selnitskiy](https://github.com/Silya-1) and [Ilya Zaharkin](https://github.com/izaharkin).

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA 

## Getting Started
### Get Access and Download data:
We tested the performance of the proposed networks on the data from Alzheimer’s Disease Neuroimaging Initiative (ADNI) project that provides a dataset of structural MRI scans. For dowload data you need to get [access](http://adni.loni.usc.edu/data-samples/access-data/). Then go to [dataset](https://ida.loni.usc.edu/home). Filter all data with this parameters:
- Weighting: 1T
- Acquision Type: 3D
- Filed Strength: 3 Tesla
- Slice Thickness: 1mm

We use this dataset to test our models’ performance for a task of classifying MRI scans of subjects with Alzheimers disease (AD), early and late mild cognitive impairment (EMCI and LMCI), and normal cohort (CN).

### Installation
- Clone this repo:
```bash
git clone https://github.com/izaharkin/mri-alzheimer
cd mri-alzheimer
``` 
- Install python 3.6 and all necessary requirements:
  - For pip users, please type the command `pip install -r requirements.txt`.

### Train/test models
- For getting good perfomance of model, prepocess data by cutting skull and run:
```bash
python3 brainiac/data_preprocessing/brain_extraction.py
```
And also you should run two notebooks: [Data Processing](./brainiac/data_preprocessing/data_processing.ipynb) and [Process Cut Data](./brainiac/data_preprocessing/process_cut_data.ipynb). 

- Train classification model:
```bash
#!./scripts/train_cyclegan.sh
python3 train.py --model ResNet152
```
- All logs and the pretrained model are saved at unique folder`./trained_model/{model}/{other parameters}` as `log{number}.log` and `model_epoch{number}.log`. To view training results and loss plots, run notebook [LogParser](LogParser.ipynb)

- For getting all changeble parameters run:
```bash
python3 train.py --help
```
For example:
  - Standart train parameters: `--num_epoch 200 --batch_size 4 --optimizer Adam --lr 3e-5 --weight_decay 1e-3`
  - Use augmntation (random rotation and noise): `--use_augmentation True`
  - Use sampling (oversampling and undersampling): `--use_sampling True --sampling_type over`
  - Apply a pre-trained model: `--use_pretrain True --path_pretrain PATH`
  
