#!/bin/bash
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/world/data-gpu-94/sysu-reid/checkpoints
# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/world/data-gpu-94/sysu-reid/checkpoints/ResNet_v1_newcode_baseline
# Where the dataset is saved to.
DATASET_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501
# WHere the log is saved to
LOG_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/log
# Wher the tfrecord file is save to
OUTPUT_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501-tfrecord/bounding_box_train
python train.py \
--learning_rate=0.01 \
--learning_rate_decay_type=fixed \
--dataset_name=Market_1501 \
--dataset_dir=${OUTPUT_DIR} \
--batch_size=32 \
--max_number_of_steps=16000 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v1_50.ckpt \
--log_dir=${LOG_DIR} \
--save_model_summary_secs=300 \
--log_every_n_steps=100 \
--optimizer=sgd \
--weight_decay=5e-4 \
--scale_height=256 \
--scale_width=128 \
--GPU_use=1

python train.py \
--learning_rate=0.001 \
--learning_rate_decay_type=fixed \
--dataset_name=Market_1501 \
--dataset_split_name=train \
--dataset_dir=${OUTPUT_DIR} \
--batch_size=32 \
--max_number_of_steps=24000 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v1_50.ckpt \
--log_dir=${LOG_DIR} \
--save_model_summary_secs=300 \
--log_every_n_steps=100 \
--optimizer=sgd \
--weight_decay=5e-4 \
--scale_height=256 \
--scale_width=128 \
--GPU_use=1
