# YouLookDifferen## YouLookDifferent
CVPR2021 supplementary materials.

*Qualitative and quantitative comparison results with more state of the art methods will be prepared in January 2020.*

## requirments:
- python == 3.5
- tensorflow == 1.13.2
- torch == 1.4.0
- trochvision == 0.5.0
- pytorch-ignite == 0.1.2

## Project_structure
Create a new folder beside the main project and rename it to `datasets_weights`. Then, download the datasets and weights from the following link and put them in this folder.

[data and weights: 3 GB](https://drive.google.com/file/d/1mfroiKvxO9IqDj2AHSew3EUqfOyLqsXT/view?usp=sharing)

## Train

### Preprocessing (you can skip this step, as we have provided the processed data at the above links):
1. extract body-keypoints and body masks
2. prepare the noneID dataset by running `Preprocessing.py`

### Train
#### Phase 1: Train STE-CNN 
1. Initialize configurations using `STE_CNN.yml`.
2. open a terminal and go to the project directory (script folder), then `$python3.5 tools/net_train.py --config_file ./configs/STE_CNN.yml`.
3. Automatically, noneID features will be extracted and saved on the disk after training is finished. However, you can do it yourself by running `$python3.5 tools/test_on_one_img.py --config_file ./configs/STE_CNN.yml`. Don't forget to update the file `STE_CNN.yml` first by declaring the path to the model weights.
#### Phase 2: Train LTE-CNN 
1. Initialize configurations using `LTE_CNN.yml`.
2. run `$python3.5 tools/net_train.py --config_file ./configs/LTE_CNN.yml`.
3. When training is finished, the model will be tested automatically. However, you can do it manually by `$python3.5 tools/net_test.py --config_file ./configs/LTE_CNN.yml`. Don't forget to update the file `LTE_CNN.yml` first by declaring the path to the model weights.

### Comparison with simple ResNet50:
1. Initialize configurations using `simple_baseline.yml`.
2. run `$python3.5 tools/net_train.py --config_file ./configs/simple_baseline.yml`.
3. update the path to the model weights in file `simple_baseline.yml` and run  `$python3.5 tools/net_test.py --config_file ./configs/simple_baseline.yml`

### Comparison with Luo et al. method:
1. Initialize configurations using `strong_baseline.yml`.
2. run `$python3.5 tools/net_train.py --config_file ./configs/strong_baseline.yml`.
3. update the path to the model weights in file `strong_baseline.yml` and run  `$python3.5 tools/net_test.py --config_file ./configs/strong_baseline.yml`
t
