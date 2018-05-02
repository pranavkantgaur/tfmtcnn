
This work is used to reproduce MTCNN, a Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks usng TensorFlow framework.

See data/WIDER_Face/README.md for downloading WIDER Face dataset.

See data/LFW_Landmark/README.md for downloading LFW facial landmark dataset.

See data/CelebA/README.md for downloading CelebA facial landmark dataset.


1) Generate a basic dataset i.e. PNet dataset.

python generate_simple_dataset.py --annotation_image_dir=./data/WIDER_Face/WIDER_train/images --annotation_file_name=./data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt --landmark_image_dir=./data/LFW_Landmark --landmark_file_name=./data/LFW_Landmark/trainImageList.txt --base_number_of_images=250000 --target_root_dir=./data/datasets/mtcnn 

2) Train PNet.

python train_model.py --network_name=PNet --train_root_dir=./data/models/mtcnn/train --dataset_root_dir=./data/datasets/mtcnn --base_learning_rate=0.01 --max_number_of_epoch=30

3) Generate a hard dataset i.e. RNet dataset.

python generate_hard_dataset.py --network_name=RNet --train_root_dir=./data/models/mtcnn/train --annotation_image_dir=./data/WIDER_Face/WIDER_train/images --annotation_file_name=./data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt --landmark_image_dir=./data/LFW_Landmark --landmark_file_name=./data/LFW_Landmark/trainImageList.txt --target_root_dir=./data/datasets/mtcnn 

4) Train RNet.

python train_model.py --network_name=RNet --train_root_dir=./data/models/mtcnn/train --dataset_root_dir=./data/datasets/mtcnn --base_learning_rate=0.01 --max_number_of_epoch=22

5) Generate a hard dataset i.e. ONet dataset.

python generate_hard_dataset.py --network_name=ONet --train_root_dir=./data/models/mtcnn/train --annotation_image_dir=./data/WIDER_Face/WIDER_train/images --annotation_file_name=./data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt --landmark_image_dir=./data/LFW_Landmark --landmark_file_name=./data/LFW_Landmark/trainImageList.txt --target_root_dir=./data/datasets/mtcnn 

6) Train ONet.

python train_model.py --network_name=ONet --train_root_dir=./data/models/mtcnn/train --dataset_root_dir=./data/datasets/mtcnn --base_learning_rate=0.01 --max_number_of_epoch=22

7) Webcamera demo.
   
python webcamera_demo.py

OR

python webcamera_demo.py --test_mode
