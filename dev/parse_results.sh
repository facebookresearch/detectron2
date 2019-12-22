#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# A shell script that parses metrics from the log file.
# Make it easier for developers to track performance of models.

LOG="$1"

if [[ -z "$LOG" ]]; then
	echo "Usage: $0 /path/to/log/file"
	exit 1
fi

# [12/15 11:47:32] trainer INFO: Total training time: 12:15:04.446477 (0.4900 s / it)
# [12/15 11:49:03] inference INFO: Total inference time: 0:01:25.326167 (0.13652186737060548 s / img per device, on 8 devices)
# [12/15 11:49:03] inference INFO: Total inference pure compute time: .....

# training time
trainspeed=$(grep -o 'Overall training.*' "$LOG" | grep -Eo '\(.*\)' | grep -o '[0-9\.]*')
echo "Training speed: $trainspeed s/it"

# inference time: there could be multiple inference during training
inferencespeed=$(grep -o 'Total inference pure.*' "$LOG" | tail -n1 | grep -Eo '\(.*\)' | grep -o '[0-9\.]*' | head -n1)
echo "Inference speed: $inferencespeed s/it"

# [12/15 11:47:18] trainer INFO: eta: 0:00:00  iter: 90000  loss: 0.5407 (0.7256)  loss_classifier: 0.1744 (0.2446)  loss_box_reg: 0.0838 (0.1160)  loss_mask: 0.2159 (0.2722)  loss_objectness: 0.0244 (0.0429)  loss_rpn_box_reg: 0.0279 (0.0500)  time: 0.4487 (0.4899)  data: 0.0076 (0.0975) lr: 0.000200  max mem: 4161
memory=$(grep -o 'max[_ ]mem: [0-9]*' "$LOG" | tail -n1 | grep -o '[0-9]*')
echo "Training memory: $memory MB"

echo "Easy to copypaste:"
echo "$trainspeed","$inferencespeed","$memory"

echo "------------------------------"

# [12/26 17:26:32] engine.coco_evaluation: copypaste: Task: bbox
# [12/26 17:26:32] engine.coco_evaluation: copypaste: AP,AP50,AP75,APs,APm,APl
# [12/26 17:26:32] engine.coco_evaluation: copypaste: 0.0017,0.0024,0.0017,0.0005,0.0019,0.0011
# [12/26 17:26:32] engine.coco_evaluation: copypaste: Task: segm
# [12/26 17:26:32] engine.coco_evaluation: copypaste: AP,AP50,AP75,APs,APm,APl
# [12/26 17:26:32] engine.coco_evaluation: copypaste: 0.0014,0.0021,0.0016,0.0005,0.0016,0.0011

echo "COCO Results:"
num_tasks=$(grep -o 'copypaste:.*Task.*' "$LOG" | sort -u | wc -l)
# each task has 3 lines
grep -o 'copypaste:.*' "$LOG" | cut -d ' ' -f 2- | tail -n $((num_tasks * 3))
