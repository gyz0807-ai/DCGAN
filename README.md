# DCGAN

## Preparation
Download four MNIST datasets from http://yann.lecun.com/exdb/mnist and put the dataset into ./mnist_data folder.

## How to use
Train on the MNIST dataset
```sh
python main.py --data_nm mnist --input_img_size 28,28,1
```
Train on the celeba dataset
```sh
python main.py --data_nm celeba --input_img_size 218,178,3
```
Generate samples
```sh
python generate_samples.py --model_path /path/to/trained/model/ --data_nm mnist/celeba
```

## MNIST reseults
<img src="https://github.com/gyz0807-ai/DCGAN/blob/master/results/sample_epoch6.png" width="50%" height="50%">

## Celeba results
<img src="https://github.com/gyz0807-ai/DCGAN/blob/master/results/celeba_sample.png" width="50%" height="50%">
