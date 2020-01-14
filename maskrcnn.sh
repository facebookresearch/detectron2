#! /bin/bash
# usage: 
# bash maskrcnn.sh 1 2 0.0025 120 20
MODEL="maskrcnn"
gpu_num=${1:-1}
total_batch_size=${2:-2}
lr=${3:-"0.0025"}
iter_num=${4:-120}
csv_frequency=${5:-20}


LOGFILE_FLODER=benchmark_log/${MODEL}_${total_batch_size}
mkdir -p $LOGFILE_FLODER


config_file="new_configs/mask_rcnn_R_50_FPN_1x.yaml"
pretrain_model="/dataset/kubernetes/dataset/models/maskrcnn/pytorch/detectron2/ImageNetPretrained/MSRA/R-50.pkl"


# remenber to change dataset in detectron2/data/datasets/builtin.py
# detectron
python3 tools/train_net.py \
    --num-gpus=${gpu_num} \
    --config-file ${config_file} \
	DATASETS.TRAIN '("coco_2017_train",)' \
	INPUT.MIN_SIZE_TRAIN "(800,)" \
        MODEL.WEIGHTS ${pretrain_model} \
        SOLVER.IMS_PER_BATCH ${total_batch_size} \
        SOLVER.BASE_LR ${lr} \
        SOLVER.MAX_ITER ${iter_num} \
        SOLVER.CHECKPOINT_PERIOD 10000 \
        OUTPUT_DIR ${LOGFILE_FLODER} \
	LOSS_PRINT_FREQUENCE 1 \
	CSV_PRINT_FREQUENCE ${csv_frequency} \
        TEST.EVAL_PERIOD 100000

