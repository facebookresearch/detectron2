python tools/train_net.py \
	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \
    SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025

#测试的时候生成json文件
python demo/demo.py --config-file /output/config.yaml  --input test.txt --output test_img/  --opts MODEL.WEIGHTS /output/model_final.pth 
