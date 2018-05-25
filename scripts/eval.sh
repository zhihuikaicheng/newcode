FEATURE_PATH=/home/yuanziyi/newPCB/RPP/39881

python eval.py \
--label_gallery_path=${FEATURE_PATH}/test_gallery_labels.mat \
--feature_gallery_path=${FEATURE_PATH}/test_gallery_features.mat \
--label_probe_path=${FEATURE_PATH}/test_probe_labels.mat \
--feature_probe_path=${FEATURE_PATH}/test_probe_features.mat \
--cam_gallery_path=${FEATURE_PATH}/testCAM.mat \
--cam_probe_path=${FEATURE_PATH}/queryCAM.mat 
