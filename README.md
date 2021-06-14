# MD-CSDNetwork: Multi-Domain Cross Stitched Network for Deepfake Detection

This repository contains the code for our submission to IEEE International Conference on Automatic Face & Gesture Recognition, 2021. 
The project was guided by Dr. Mayank Vatsa and Dr. Richa Singh at the Image Analysis & Biometrics Lab at IIT Jodhpur.

Paper link on my private drive account - https://drive.google.com/file/d/1UzqJ4IN6B7CZLEJrB2uG1WsZsJeZsBAW/view?usp=sharing

>The rapid progress in the ease of creating and spreading ultra-realistic media over social platforms calls for an urgent need to develop a generalizable deepfake detection technique. We observe that current deepfake generation methods leave discriminative artifacts in the frequency spectrum of fake images and videos. Inspired by this observation, in this paper, we present a novel approach, termed as MD-CSDNetwork, for combining the features in the spatial and frequency domains to mine a shared discriminative representation for classifying deepfakes. MD-CSDNetwork is a novel cross-stitched network with two parallel branches carrying the spatial and frequency information, respectively. We hypothesize that these multidomain input data streams can be considered as related supervisory signals. The supervision from both branches ensures better performance and generalization. Further, the concept of cross-stitch connections is utilized where they are inserted between the two branches to learn an optimal combination of domain specific and shared representations from other domains automatically. Extensive experiments are conducted on the popular benchmark dataset FaceForeniscs++ for forgery classification. We report improvements over all the manipulation types in FaceForensics++ dataset and comparable results with state-ofthe-art methods for cross-database evaluation on the Celeb-DF
dataset and the Deepfake Detection Dataset.

## Datasets

We utilise three popular datasets:
* [FaceForensics++](https://github.com/ondyari/FaceForensics)
* [Celeb-DF](http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html)
* [Deepfake Detection](https://github.com/ondyari/FaceForensics/tree/master/dataset/DeepFakeDetection)

## Dataset Preparation

If running the code in a container, then use the yaml files in the docker folder in the repository.

``` 
Script Usage: process_frames.py [--video_path] [--output_path] [--start_frame] [--end_frame] 
to extract frames and do an enlarged face crop for all the frames. 

Arguments:
--video_path    Path to the folder containing videos
--output_path   Path to the folder for extracted frames
--start_frame   Number of start frame
--end_frame     Number of end frame

Script Usage: prepare_dataset.py [--raw] [--log] [--color] DIRECTORY

positional arguments:
DIRECTORY  Directory of images to convert into a TFRecord file

optional arguments:
--raw      Save image data as raw image or 
--log      Log scale Images.
--color    Compute as RGB image.
```
## Experiments

### Training
```
Script Usage: classifier.py train [--epochs EPOCHS]
                          [--image_size IMAGE_SIZE]
                          [--classes CLASSES]
                          [--batch_size BATCH_SIZE]
                          MODEL CKPT_PATH TRAIN_DATASET VAL_DATASET

positional arguments:
  MODEL                 Select model to train {cnn, xception}
  CKPT_PATH             Path where the train checkpoint will be stored
  TRAIN_DATASET         Train Dataset to load.
  VAL_DATASET           Validation Dataset to load.

optional arguments:
  --epochs EPOCHS
                        Epochs to train for; Default: 50.
  --image_size IMAGE_SIZE
                        Image size. Default: [128, 128, 3]
  --early_stopping EARLY_STOPPING
                        Early stopping criteria. Default: 5
  --classes CLASSES     Classes. Default: 2
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size. Default: 32

```

### Testing
```
Script Usage: classifer.py test [--image_size IMAGE_SIZE]
                         [--batch_size BATCH_SIZE]
                         CKPT_PATH TEST_DATASET

positional arguments:
  CKPT_PATH             Path to checkpoint.
  TEST_DATASET          Test dataset to load.

optional arguments:
  --image_size IMAGE_SIZE
                        Image size. Default: [128, 128, 3]
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size. Default: 32
```
