
This repository contains code for a image segmentation model based on [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) implemented in PyTorch.

## Installation
1. Create python virtual environment.
```sh
python -m venv venv
pip install -r requirements.txt
```

## Training on original dataset
Make sure to put the files as the following structure (e.g. the number of classes is 2):
```
inputs
└── <dataset name>
    ├── images
    |   ├── 0a7e06.jpg
    │   ├── 0aab0a.jpg
    │   ├── 0b1761.jpg
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 0a7e06.png
        |   ├── 0aab0a.png
        |   ├── 0b1761.png
        |   ├── ...
        |
        └── 1
            ├── 0a7e06.png
            ├── 0aab0a.png
            ├── 0b1761.png
            ├── ...
```

1. Train the model.
```
source venv/bin/activate
python main.py
```
2. Evaluate.
```
python val.py --name <dataset name>_NestedUNet_woDS
```

## Results
### DSB2018 (96x96)

Here is the results on DSB2018 dataset (96x96) with LovaszHingeLoss.

| Model                           |   IoU   |  Loss   |
|:------------------------------- |:-------:|:-------:|
| U-Net                           |  0.839  |  0.365  |
| Nested U-Net                    |  0.842  |**0.354**|
| Nested U-Net w/ Deepsupervision |**0.843**|  0.362  |
