This work is used to reproduce MTCNN, a Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks using TensorFlow framework.

See data/WIDER_Face/README.md for downloading WIDER Face dataset.

See data/CelebA/README.md for downloading CelebA facial landmark dataset.

1) Prepare CelebA dataset for input.

python prepare_celeba_dataset.py --bounding_box_file_name=../data/CelebA/list_bbox_celeba.txt --landmark_file_name=../data/CelebA/list_landmarks_celeba.txt --output_file_name=../data/CelebA/CelebA.txt 

2) Generate a basic dataset i.e. PNet dataset.

python generate_simple_dataset.py --annotation_image_dir=../data/WIDER_Face/WIDER_train/images --annotation_file_name=../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt --landmark_image_dir=../data/CelebA/images --landmark_file_name=../data/CelebA/CelebA.txt --sample_multiplier_factor=10 --target_root_dir=../data/datasets/mtcnn 

3) Train PNet.

python train_model.py --network_name=PNet --train_root_dir=../data/models/mtcnn/train --dataset_root_dir=../data/datasets/mtcnn --base_learning_rate=0.01 --max_number_of_epoch=30

4) Generate a hard dataset i.e. RNet dataset.

python generate_hard_dataset.py --network_name=RNet --train_root_dir=../data/models/mtcnn/train --annotation_image_dir=../data/WIDER_Face/WIDER_train/images --annotation_file_name=../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt --landmark_image_dir=../data/CelebA/images --landmark_file_name=../data/CelebA/CelebA.txt --target_root_dir=../data/datasets/mtcnn 

5) Train RNet.

python train_model.py --network_name=RNet --train_root_dir=../data/models/mtcnn/train --dataset_root_dir=../data/datasets/mtcnn --base_learning_rate=0.01 --max_number_of_epoch=22

6) Generate a hard dataset i.e. ONet dataset.

python generate_hard_dataset.py --network_name=ONet --train_root_dir=../data/models/mtcnn/train --annotation_image_dir=../data/WIDER_Face/WIDER_train/images --annotation_file_name=../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt --landmark_image_dir=../data/CelebA/images --landmark_file_name=../data/CelebA/CelebA.txt --target_root_dir=../data/datasets/mtcnn 

7) Train ONet.

python train_model.py --network_name=ONet --train_root_dir=../data/models/mtcnn/train --dataset_root_dir=../data/datasets/mtcnn --base_learning_rate=0.01 --max_number_of_epoch=22

8) Webcamera demo.
   
python webcamera_demo.py

9) Webcamera demo using trained models.

python webcamera_demo.py --test_mode

