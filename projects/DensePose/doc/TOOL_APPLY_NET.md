# Apply Net

`apply_net` is a tool to print or visualize DensePose results on a set of images.
It has two modes: `dump` to save DensePose model results to a pickle file
and `show` to visualize them on images.

## Dump Mode

The general command form is:
```bash
python apply_net.py dump [-h] [-v] [--output <dump_file>] <config> <model> <input>
```

There are three mandatory arguments:
 - `<config>`, configuration file for a given model;
 - `<model>`, model file with trained parameters
 - `<input>`, input image file name, pattern or folder

One can additionally provide `--output` argument to define the output file name,
which defaults to `output.pkl`.


Examples:

1. Dump results of a DensePose model with ResNet-50 FPN backbone for images
   in a folder `images` to file `dump.pkl`:
```bash
python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml DensePose_ResNet50_FPN_s1x-e2e.pkl images --output dump.pkl -v
```

2. Dump results of a DensePose model with ResNet-50 FPN backbone for images
   with file name matching a pattern `image*.jpg` to file `results.pkl`:
```bash
python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml DensePose_ResNet50_FPN_s1x-e2e.pkl "image*.jpg" --output results.pkl -v
```



If you want to load the pickle file of DensePose model results: 
```
f = open('/your_result_path/result.pkl', 'rb')
data = pickle.load(f)
```


The pickle loading code should be run in the `/your_detectron2_path/detectron2_repo/projects/DensePose/` directory, 
or `ModuleNotFoundError: No module named 'densepose'` will occur. 

If you have to run the code in other directory, simply add this before pickle loading:
```
import sys
sys.path.append("/your_detectron2_path/detectron2_repo/projects/DensePose/")
```

The file `result.pkl` contain the list of results per image, for each image the result is a dictionary. 
DensePose results are stored under `'pred_densepose'` key, which is an instance of DensePoseResult class. 
```
data: [{'file_name': '/your_path/image1.jpg', 
        'scores': tensor([0.9884]), 
        'pred_boxes_XYXY': tensor([[ 69.6114,   0.0000, 706.9797, 706.0000]]), 
        'pred_densepose': <densepose.structures.DensePoseResult object at 0x7f791b312470>}, 
       {'file_name': '/your_path/image2.jpg', 
        'scores': tensor([0.9999, 0.5373, 0.3991]), 
        'pred_boxes_XYXY': tensor([[ 59.5734,   7.7535, 579.9311, 932.3619],
                                   [612.9418, 686.1254, 612.9999, 704.6053],
                                   [164.5081, 407.4034, 598.3944, 920.4266]]), 
        'pred_densepose': <densepose.structures.DensePoseResult object at 0x7f7071229be0>},
       {'file_name': '/your_path/image3.jpg', 
        'scores': tensor([0.9987, 0.8648, 0.7277, 0.6785]), 
        'pred_boxes_XYXY': tensor([[ 29.5393,  29.7294, 260.2740, 929.7100],
                                   [ 25.8856,  34.7234, 272.5084, 419.4978],
                                   [ 47.4713, 475.4232, 245.2225, 906.9631],
                                   [ 84.8272, 241.3668, 128.0741, 301.6587]]),
        'pred_densepose': <densepose.structures.DensePoseResult object at 0x7f791b3126d8>}]

```

The `'pred_densepose'` object is encoded as a string, 
thus you can use `DensePoseResult.decode_png_data()` to decode the string and extract the IUV array. 
```
img_id = 2  # Which image to extract IUV array from

densepose_result = data[img_id]['pred_densepose']
for i, result_encoded_w_shape in enumerate(densepose_result.results):
  if i != 0:
    break
  iuv_arr = DensePoseResult.decode_png_data(*result_encoded_w_shape)
  bbox_xywh = densepose_result.boxes_xywh[i]
print('iuv arr:', iuv_arr)  #IUV_array is extracted
print('iuv arr shape:', iuv_arr.shape) 
print('bbox_xywh:', bbox_xywh)
```
For each image, `densepose_result.results` may contain several iuv_arrays, here we choose the first one which has the highest detection score as the IUV array. 





## Visualization Mode

The general command form is:
```bash
python apply_net.py show [-h] [-v] [--min_score <score>] [--nms_thresh <threshold>] [--output <image_file>] <config> <model> <input> <visualizations>
```

There are four mandatory arguments:
 - `<config>`, configuration file for a given model;
 - `<model>`, model file with trained parameters
 - `<input>`, input image file name, pattern or folder
 - `<visualizations>`, visualizations specifier; currently available visualizations are:
   * `bbox` - bounding boxes of detected persons;
   * `dp_segm` - segmentation masks for detected persons;
   * `dp_u` - each body part is colored according to the estimated values of the
     U coordinate in part parameterization;
   * `dp_v` - each body part is colored according to the estimated values of the
     V coordinate in part parameterization;
   * `dp_contour` - plots contours with color-coded U and V coordinates


One can additionally provide the following optional arguments:
 - `--min_score` to only show detections with sufficient scores that are not lower than provided value
 - `--nms_thresh` to additionally apply non-maximum suppression to detections at a given threshold
 - `--output` to define visualization file name template, which defaults to `output.png`.
   To distinguish output file names for different images, the tool appends 1-based entry index,
   e.g. output.0001.png, output.0002.png, etc...


The following examples show how to output results of a DensePose model
with ResNet-50 FPN backbone using different visualizations for image `image.jpg`:

1. Show bounding box and segmentation:
```bash
python apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml DensePose_ResNet50_FPN_s1x-e2e.pkl image.jpg bbox,dp_segm -v
```
![Bounding Box + Segmentation Visualization](images/res_bbox_dp_segm.jpg)

2. Show bounding box and estimated U coordinates for body parts:
```bash
python apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml DensePose_ResNet50_FPN_s1x-e2e.pkl image.jpg bbox,dp_u -v
```
![Bounding Box + U Coordinate Visualization](images/res_bbox_dp_u.jpg)

3. Show bounding box and estimated V coordinates for body parts:
```bash
python apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml DensePose_ResNet50_FPN_s1x-e2e.pkl image.jpg bbox,dp_v -v
```
![Bounding Box + V Coordinate Visualization](images/res_bbox_dp_v.jpg)

4. Show bounding box and estimated U and V coordinates via contour plots:
```bash
python apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml DensePose_ResNet50_FPN_s1x-e2e.pkl image.jpg dp_contour,bbox -v
```
![Bounding Box + Contour Visualization](images/res_bbox_dp_contour.jpg)
