# Activity Recognition in Childrenwith Autism-Related Behaviours
![Loading Framework](data/framework.png "Framework overview")

Autism Spectrum Disorder (ASD), known as autism, is a lifelong developmental disorder that affects most children around the world. Analysing autism-related behaviours is a common way to diagnose ASD. However, this diagnosis process is time-consuming due to long-term behaviour observation and the scarce availability of specialists. Here, we propose a regain-based computer vision system to help clinicians and parents analyse children’s behaviours. We combinate and collected a dataset for analysing autism-related actions based on videos of children in an uncontrolled environment. Then pre-processed this dataset by cropping the target child from a video using Detectron2 person detection model to reduce the noise environment. We propose an approach that leverages current state-of-the-art CNN models to extract action features from videos per frame and utilizes temporal models to classify autism-related behaviours by analysing the relationship between frames within a video. The proposed method achieved 0.87 accuracy (F1-score 0.87) to classify the three actions in the collected dataset. The experimental results demonstrate the potential ability of our proposed model to reliable and accurately help clinicians diagnose ASD order and show the video-based model can efficiently speed up the diagnose process of ASD.
### Dependencies
* Python >= 3.8
* [PyTorch](https://pytorch.org) >= 1.8
* [kinetics-i3d](https://github.com/deepmind/kinetics-i3d) (if you want to use i3d feature extractor)
* [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) or `pip install efficientnet_pytorch`

### Datasets
Self-Stimulatory Behaviour Dataset (SSBD) is the only one publicly available dataset. The dataset’s videos are in nature and recorded in an uncontrolled environment [1]. Three actions in this dataset: Armflapping, Headbanging, and Spinning. The total actual videos reported in original paper was 75 videos, in    which only 60 are downloadable due to privacy concerns of Youtube. In this dataset, some of videos are too noisy (almost in a very dark environment). We collected a few new videos from Youtube to replace these noisy videos in the dataset. In our new dataset, there are total 61 videos. Here is the detail of this new dataset: 

|           |*Armflapping*|*Headbanging*|*Spinning*|
|:-----------------:|:--------:|:----------:|:-----------:|
| `No. Videos in our new dataset ` |   20   |    21    |      20      |
### Feature extractor
There are some feature extractors that you can use 
1. EfficientNet (Remember to install EfficientNet package)
2. MobileNet (Aleardy in this repository). This package is from https://github.com/d-li14/mobilenetv3.pytorch.
3. [kinetics-i3d](https://github.com/deepmind/kinetics-i3d). Due to license issues, you need to download by yourself and put this package under `extractor_models` folder. And then you can use `i3d_extractor.py` to extract features. 

### Training
#### with iamge features
1. Sorry. Due to ethical issues, I couldn't public the extracted features by i3d. Please use your own features (see 2.). In default, you can used my extracted features in `data/i3d_feature`. The features were extracted by I3D model that pre-trained on Kinetice dataset.  
  Run with `python train_with_features_tcn.py`（for TCN model） or `python train_with_features_ms_tcn.py` (for MS-TCN model) 
  Then, the trained model parameters will be saved in `model_zoo/your_model_zoo`
2. Using your own features 
   Put your features file under `data/other_features`. And change the command based on the instructions below:
   ![Loading Command](data/command.png "Command")
   OR: 
   ![Loading Command](data/command1.png "Command")
#### with images 
Here is a example of using `EfficientNetB3+TCN` model. 
1. Put your images under `data/images`.  
2. run `python train_with_images.py`. And you can modify the command based on the fellowing instructions:
![Loading Command](data/command2.png "Command")
3. The trained model's parameters would be saved under `model_zoo/your_model_zoo`.
If you want to use different model, just changed this line to your desired model.  
![Loading Command](data/command3.png "Command")

