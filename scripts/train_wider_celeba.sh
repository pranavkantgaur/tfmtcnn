echo '########################################################################################################################################################################################################'

if [ "$MTCNN_ROOT_DIR" == "" ]; then
	export MTCNN_ROOT_DIR=/
fi

echo 'Check MTCNN_ROOT_DIR, current MTCNN_ROOT_DIR is - '$MTCNN_ROOT_DIR

echo '########################################################################################################################################################################################################'

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=$MTCNN_ROOT_DIR/git-space/tfmtcnn:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

echo '########################################################################################################################################################################################################'

cd $MTCNN_ROOT_DIR/git-space/tfmtcnn/tfmtcnn

echo 'Prepare CelebA dataset for input.'

python tools/prepare_celeba_dataset.py --bounding_box_file_name=../data/CelebA/list_bbox_celeba.txt --landmark_file_name=../data/CelebA/list_landmarks_celeba.txt --output_file_name=../data/CelebA/CelebA.txt

echo 'Prepared CelebA dataset for input.'

echo '########################################################################################################################################################################################################'

echo 'Generate a basic dataset i.e. PNet dataset.'

python generate_simple_dataset.py --annotation_image_dir=../data/WIDER_Face/WIDER_train/images --annotation_file_name=../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt --landmark_image_dir=../data/CelebA/images --landmark_file_name=../data/CelebA/CelebA.txt --base_number_of_images=700000 --target_root_dir=../data/datasets/mtcnn

echo 'Generated a basic dataset i.e. PNet dataset.'

echo '########################################################################################################################################################################################################'

echo 'Train PNet.'

python train_model.py --network_name=PNet --train_root_dir=../data/models/mtcnn/train --dataset_root_dir=../data/datasets/mtcnn --base_learning_rate=0.01 --max_number_of_epoch=36

echo 'Trained PNet.'

echo '########################################################################################################################################################################################################'

echo 'Generate a hard dataset i.e. RNet dataset.'

python generate_hard_dataset.py --network_name=RNet --train_root_dir=../data/models/mtcnn/train --annotation_image_dir=../data/WIDER_Face/WIDER_train/images --annotation_file_name=../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt --landmark_image_dir=../data/CelebA/images --landmark_file_name=../data/CelebA/CelebA.txt --base_number_of_images=500000 --target_root_dir=../data/datasets/mtcnn 

echo 'Generated a hard dataset i.e. RNet dataset.'

echo '########################################################################################################################################################################################################'

echo 'Train RNet.'

python train_model.py --network_name=RNet --train_root_dir=../data/models/mtcnn/train --dataset_root_dir=../data/datasets/mtcnn --base_learning_rate=0.01 --max_number_of_epoch=36

echo 'Trained RNet.'

echo '########################################################################################################################################################################################################'

echo 'Generate a hard dataset i.e. ONet dataset.'

python generate_hard_dataset.py --network_name=ONet --train_root_dir=../data/models/mtcnn/train --annotation_image_dir=../data/WIDER_Face/WIDER_train/images --annotation_file_name=../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt --landmark_image_dir=../data/CelebA/images --landmark_file_name=../data/CelebA/CelebA.txt --base_number_of_images=500000 --target_root_dir=../data/datasets/mtcnn 

echo 'Generated a hard dataset i.e. ONet dataset.'

echo '########################################################################################################################################################################################################'

echo 'Train ONet.'

python train_model.py --network_name=ONet --train_root_dir=../data/models/mtcnn/train --dataset_root_dir=../data/datasets/mtcnn --base_learning_rate=0.01 --max_number_of_epoch=36

echo 'Trained ONet.'

echo '########################################################################################################################################################################################################'


