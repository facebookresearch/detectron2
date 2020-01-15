set -x
rm -rf output
python3 tools/train_net.py \
    --config-file golden_standard/mrcn_1x_train.yaml
