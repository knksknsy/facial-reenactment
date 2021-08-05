# Facial Reenactment Creation and Detection

This project was developed during the course of my master's thesis "Machine Learning Methods for Facial Reenactment Creation and Detection". This framework combines a generative and a discriminative model to create and detect facial reenactments.

## Getting Started

### Dependencies

* PyTorch v1.7.1
* Python v3.8.5
* NVIDIA CUDA Toolkit v10.1
* cuDNN v7.6.5

* Install dependency packages:
```
$ conda install --file requirements.txt
```

### Download Datasets
* [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
* [FaceForensics++](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform)
* or facial dataset of choice

### Pretrained Models
[Download](https://drive.google.com/drive/folders/1529CFUjnNGYuloE16MwrYZwgLISGBv25?usp=sharing)

### FR Creation Inference
#### Grayscale Model 7
```
$ python run_creation.py infer  \
                --device cuda   \
                --source <PATH> \   # Path to source image or video
                --target <PATH> \   # Path to target image
                --model /media/models/creation/exp7/Generator_t20210311_2251_e029_i00037500.pth \
                --config ./configs/config_creation_model7.yaml
```
#### RGB Model 11
```
$ python run_creation.py infer  \
                --device cuda   \
                --source <PATH> \   # Path to source image or video
                --target <PATH> \   # Path to target image
                --model /media/models/creation/exp11/Generator_t20210401_1236_e029_i00150000.pth \
                --config ./configs/config_creation_model11.yaml
```
### FR Detection Inference
#### Model 6
```
$ python run_detection.py infer \
                --device cuda   \
                --source <PATH> \   # Path to image or video
                --model /media/models/detection/exp6/SiameseResNet_t20210416_2034_e019_i00010710.pth \
                --config ./configs/config_detection_model6.yaml
```

### Executing FR creation program

* Create facial dataset with facial landmarks extraction:
```
$ python run_creation.py dataset \
                            --source <PATH>         # Path to raw video files
                            --output <PATH>         # Output path for extracted images and landmarks
                            --csv <PATH>            # Path to CSV containing data pairs for training or testing
                            --num_videos [0,N]      # Limit number of frames to be extracted; 0 = no limit
                            --image_size_db <N>     # Output shape of extracted images
                            --device <cuda, cpu>
                            --config <PATH>         # Path to YAML config file for further settings (see ./configs/config.yaml)
```

* Start training FR creation model:
```
$ python run_creation.py train \
                --num_workers <N>
                --device <cuda, cpu>
                --dataset_train <PATH>  # Path to extracted training set
                --dataset_test <PATH>   # Path to extracted testing set 
                --csv_train <PATH>      # Path to CSV containing training pairs
                --csv_test <PATH>       # Path to CSV containing testing pairs
                --test                  # Test model after each epoch
                --seed <N>              # Seed for random number generators
                --config <PATH>         # Path to YAML config file for further settings (see ./configs/config.yaml)
                --plots <PATH>          # Path to JSON plot config file (see ./configs/plots.json)
```

* Inference of FR creation model:
```
$ python run_creation.py infer \
                --device <cuda, cpu>
                --source <PATH>         # Path to source image or video
                --target <PATH>         # Path to target image
                --model <PATH>          # Path to FR creation generator model
                --config <PATH>         # Path to YAML config file for further settings (see ./configs/config.yaml)
```

### Executing FR detection program

* Create facial dataset with facial landmarks extraction:
```
$ python run_detection.py dataset \
                            --source <PATH>         # Path to raw video files
                            --output <PATH>         # Output path for extracted images and landmarks
                            --csv <PATH>            # Path to CSV containing data pairs for training or testing
                            --num_videos [0,N]      # Limit number of frames to be extracted; 0 = no limit
                            --image_size_db <N>     # Output shape of extracted images
                            --device <cuda, cpu>
                            --config <PATH>         # Path to YAML config file for further settings (see ./configs/config.yaml)
```

* Start training FR detection model:
```
$ python run_detection.py train \
                --num_workers <N>
                --device <cuda, cpu>
                --dataset_train <PATH>  # Path to extracted training set
                --dataset_test <PATH>   # Path to extracted testing set 
                --csv_train <PATH>      # Path to CSV containing training pairs
                --csv_test <PATH>       # Path to CSV containing testing pairs
                --test                  # Test model after each epoch
                --seed <N>              # Seed for random number generators
                --config <PATH>         # Path to YAML config file for further settings (see ./configs/config.yaml)
                --plots <PATH>          # Path to JSON plot config file (see ./configs/plots.json)
```

* Inference of FR detection model:
```
$ python run_detection.py infer \
                --device <cuda, cpu>
                --source <PATH>         # Path to image or video
                --model <PATH>          # Path to FR detection model
                --config <PATH>         # Path to YAML config file for further settings (see ./configs/config.yaml)
```
