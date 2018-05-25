#!/bin/bash
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/world/data-gpu-94/sysu-reid/checkpoints
# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/world/data-gpu-94/sysu-reid/checkpoints/ResNet_v1_PCBRPP_new4
# Where the dataset is saved to.
DATASET_DIR=/home/zhangkaicheng/Market-1501
# WHere the log is saved to
LOG_DIR=/home/yuanziyi/log
# Wher the tfrecord file is save to
PROBE_OUTPUT_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501-tfrecord/query
# Wher the tfrecord file is save to
GALLERY_OUTPUT_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501-tfrecord/bounding_box_test

python get_features_rpp.py \
--dataset_name=Market_1501 \
--probe_dataset_dir=${PROBE_OUTPUT_DIR} \
--gallery_dataset_dir=${GALLERY_OUTPUT_DIR} \
--batch_size=32 \
--max_number_of_steps=10001 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v1_50.ckpt \
--log_dir=${LOG_DIR} \
--weight_decay=5e-4 \
--ckpt_num=39881 \
--scale_height=384 \
--scale_width=128 \
--GPU_use=6 \
--only_pcb=True \
--only_classifier=False \
--max_step_to_train_pcb=24000 \
--max_step_to_train_classifier=4000

# python get_features_rpp.py \
# --dataset_name=Market_1501 \
# --probe_dataset_dir=${PROBE_OUTPUT_DIR} \
# --gallery_dataset_dir=${GALLERY_OUTPUT_DIR} \
# --batch_size=32 \
# --max_number_of_steps=10001 \
# --checkpoint_dir=${TRAIN_DIR} \
# --pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v1_50.ckpt \
# --log_dir=${LOG_DIR} \
# --weight_decay=5e-4 \
# --ckpt_num=38742 \
# --scale_height=384 \
# --scale_width=128 \
# --GPU_use=6 \
# --only_pcb=True \
# --only_classifier=False \
# --max_step_to_train_pcb=24000 \
# --max_step_to_train_classifier=4000

# python get_features_rpp.py \
# --dataset_name=Market_1501 \
# --probe_dataset_dir=${PROBE_OUTPUT_DIR} \
# --gallery_dataset_dir=${GALLERY_OUTPUT_DIR} \
# --batch_size=32 \
# --max_number_of_steps=10001 \
# --checkpoint_dir=${TRAIN_DIR} \
# --pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v1_50.ckpt \
# --log_dir=${LOG_DIR} \
# --weight_decay=5e-4 \
# --ckpt_num=37603 \
# --scale_height=384 \
# --scale_width=128 \
# --GPU_use=6 \
# --only_pcb=True \
# --only_classifier=False \
# --max_step_to_train_pcb=24000 \
# --max_step_to_train_classifier=4000