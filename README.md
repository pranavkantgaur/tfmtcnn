# MTCNN using TensorFlow framework
This work is used to reproduce MTCNN, a Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks using TensorFlow framework.
  - See <MTCNN_ROOT>/data/WIDER_Face/README.md for downloading WIDER Face dataset.
  - See <MTCNN_ROOT>/data/CelebA/README.md for downloading CelebA facial landmark dataset.

## Prepare CelebA dataset for input.

```sh
python tfmtcnn/tfmtcnn/tools/prepare_celeba_dataset.py \
    --bounding_box_file_name ../data/CelebA/list_bbox_celeba.txt \
    --landmark_file_name ../data/CelebA/list_landmarks_celeba.txt \
    --output_file_name ../data/CelebA/CelebA.txt 
```
## Generate a basic dataset i.e. PNet dataset.

```sh
python tfmtcnn/tfmtcnn/generate_simple_dataset.py \
	--annotation_image_dir ../data/WIDER_Face/WIDER_train/images \ 
	--annotation_file_name ../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt \
	--landmark_image_dir ../data/CelebA/images \
	--landmark_file_name ../data/CelebA/CelebA.txt \
	--base_number_of_images 700000 \
	--target_root_dir ../data/datasets/mtcnn 
```	

## Train PNet.

```sh
python tfmtcnn/tfmtcnn/train_model.py \
	--network_name PNet \ 
	--train_root_dir ../data/models/mtcnn/train \
	--dataset_root_dir ../data/datasets/mtcnn \
	--base_learning_rate 0.01 \
	--learning_rate_epoch 8 16 24 \
	--max_number_of_epoch 32
```

## Generate a hard dataset i.e. RNet dataset.

```sh
python tfmtcnn/tfmtcnn/generate_hard_dataset.py \
	--network_name RNet \ 
	--train_root_dir ../data/models/mtcnn/train \
	--annotation_image_dir ../data/WIDER_Face/WIDER_train/images \ 
	--annotation_file_name ../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt \
	--landmark_image_dir ../data/CelebA/images \
	--landmark_file_name ../data/CelebA/CelebA.txt \
	--base_number_of_images 700000 \
	--target_root_dir ../data/datasets/mtcnn 
```	

## Train RNet.

```sh
python tfmtcnn/tfmtcnn/train_model.py \
	--network_name RNet \ 
	--train_root_dir ../data/models/mtcnn/train \
	--dataset_root_dir ../data/datasets/mtcnn \
	--base_learning_rate 0.01 \
	--learning_rate_epoch 8 16 24 \
	--max_number_of_epoch 32
```

## Generate a hard dataset i.e. ONet dataset.

```sh
python tfmtcnn/tfmtcnn/generate_hard_dataset.py \
	--network_name ONet \ 
	--train_root_dir ../data/models/mtcnn/train \
	--annotation_image_dir ../data/WIDER_Face/WIDER_train/images \ 
	--annotation_file_name ../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt \
	--landmark_image_dir ../data/CelebA/images \
	--landmark_file_name ../data/CelebA/CelebA.txt \
	--base_number_of_images 700000 \
	--target_root_dir ../data/datasets/mtcnn 
```	

## Train ONet.

```sh
python tfmtcnn/tfmtcnn/train_model.py \
	--network_name ONet \ 
	--train_root_dir ../data/models/mtcnn/train \
	--dataset_root_dir ../data/datasets/mtcnn \
	--base_learning_rate 0.01 \
	--learning_rate_epoch 8 16 24 \
	--max_number_of_epoch 32
```

## Webcamera demo.
```sh
python  tfmtcnn/tfmtcnn/webcamera_demo.py
```

## Webcamera demo using trained models.
```sh
python  tfmtcnn/tfmtcnn/webcamera_demo.py --test_mode
```

## Evaluate the model accuracy on the FDDB dataset.
```sh
python tfmtcnn/tfmtcnn/evaluate_model.py \
	--model_root_dir tfmtcnn/tfmtcnn/models/mtcnn/train \
	--dataset_name FDDBDataset \
	--annotation_image_dir /datasets/FDDB/ \ 
	--annotation_file_name /datasets/FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt
```

