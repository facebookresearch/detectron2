import itertools

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator


class WandbVisualizer(DatasetEvaluator):
    """
    WandbVisualizer logs the model predictions over time in form of interactive W&B Tables.
    Sopported tasks: Bounding Box Detection, Semantic Segmentation
    """

    def __init__(self, dataset_name, size=50) -> None:
        """
        Args:
            dataset_name (str): Name of the registered dataset to be Visualized
            size (int): Maximum number of data-points/rows to be visualized
        """
        super().__init__()

        import wandb

        self.wandb = wandb
        self.dataset_name = dataset_name
        self.size = size
        self._run = None
        # Table logging utils
        self._evalset_table_rows = []  # reset after log operation
        self._evalset_table_ref = None  # eval artifact refrence for deduping. Don't reset
        self._map_table_row_file_name = {}
        self._table_thing_classes = None
        self._table_stuff_classes = None

        # parsed metadata
        self.thing_class_names = []
        self.thing_index_to_class = {}
        self.stuff_class_names = []
        self.stuff_index_to_class = {}

    def process(self, inputs, outputs):
        if self.size > 0 and len(self._evalset_table_rows) >= self.size // (comm.get_world_size()):
            return
        parsed_output = self._parse_prediction(outputs[0])
        parsed_output["file_name"] = inputs[0].get("file_name")
        table_row = self._plot_table_row(parsed_output)
        self._evalset_table_rows.append(table_row)

    def evaluate(self):
        comm.synchronize()
        table_rows = comm.gather(self._evalset_table_rows)
        if comm.is_main_process():
            if self.wandb.run is None:
                raise Exception(
                    "wandb run is not initialized. Either call wandb.init(...) or set WandbWriter()"
                )
            self._run = self.wandb.run

            table_rows = list(itertools.chain(*table_rows))
            _evalset_table = self._build_evalset_table(table_rows[-1][1])
            for _row_idx, table_row in enumerate(table_rows):
                # use reference of table if present
                if self._evalset_table_ref is None:
                    _evalset_table.add_data(*table_row)
                    self._map_table_row_file_name[table_row[0]] = _row_idx
                    _row_idx = _row_idx + 1
                else:
                    row_idx = self._map_table_row_file_name[table_row[0]]
                    ref_table_row = self._evalset_table_ref[row_idx]
                    _evalset_table.add_data(
                        table_row[0],
                        self.wandb.Image(
                            ref_table_row[1], boxes=table_row[1]._boxes, masks=table_row[1]._masks
                        ),
                        *table_row[2:],
                    )

            self._run.log({self.dataset_name: _evalset_table})
            if self._evalset_table_ref is None:
                self._use_table_as_artifact(_evalset_table)
        return super().evaluate()

    def reset(self):
        self._build_dataset_metadata()
        self._evalset_table_rows = []
        return super().reset()

    def _build_dataset_metadata(self):
        """
        Builds parsed metadata lists and mappings dicts to facilitate logging.
        Builds a list of metadata for each of the validation dataloaders.

        E.g.
        # set the properties separately
        MetadataCatalog.get("val1").property = ...
        MetadataCatalog.get("val2").property = ...

        Builds metadata objects for stuff and thing classes:
        1. self.thing_class_names/self.stuff_class_names -- List[str] of category names
        2. self.thing_index_to_class/self.stuff_index_to_class-- Dict[int, str]

        """
        meta = MetadataCatalog.get(self.dataset_name)

        # Parse thing_classes
        if hasattr(meta, "thing_classes"):
            self.thing_class_names = meta.thing_classes

        wandb_thing_classes = []
        # NOTE: The classs indeces starts from 1 instead of 0.
        for i, name in enumerate(self.thing_class_names, 1):
            self.thing_index_to_class[i] = name
            wandb_thing_classes.append({"id": i, "name": name})

        self._table_thing_classes = self.wandb.Classes(wandb_thing_classes)

        # Parse stuff_classes
        if hasattr(meta, "stuff_classes"):
            self.stuff_class_names = meta.stuff_classes

        wandb_stuff_classes = []
        for i, name in enumerate(self.stuff_class_names):
            self.stuff_index_to_class[i] = name
            wandb_stuff_classes.append({"id": i, "name": name})

        self._table_stuff_classes = self.wandb.Classes(wandb_stuff_classes)

    def _parse_prediction(self, pred):
        """
        Parse prediction of one image and return the primitive martices to plot wandb media files.
        Moves prediction from GPU to system memory.

        Args:
            pred (detectron2.structures.instances.Instances): Prediction instance for the image

        returns:
            Dict (): parsed predictions
        """
        parsed_pred = {}
        if pred.get("instances") is not None:
            pred_ins = pred["instances"]
            pred_ins = pred_ins[pred_ins.scores > 0.7]
            parsed_pred["boxes"] = (
                pred_ins.pred_boxes.tensor.tolist() if pred_ins.has("pred_boxes") else None
            )
            parsed_pred["classes"] = (
                pred_ins.pred_classes.tolist() if pred_ins.has("pred_classes") else None
            )
            parsed_pred["scores"] = pred_ins.scores.tolist() if pred_ins.has("scores") else None
            parsed_pred["pred_masks"] = (
                pred_ins.pred_masks.cpu().detach().numpy() if pred_ins.has("pred_masks") else None
            )  # wandb segmentation panel supports np
            parsed_pred["pred_keypoints"] = (
                pred_ins.pred_keypoints.tolist() if pred_ins.has("pred_keypoints") else None
            )

        if pred.get("sem_seg") is not None:
            parsed_pred["sem_mask"] = pred["sem_seg"].argmax(0).cpu().detach().numpy()

        if pred.get("panoptic_seg") is not None:
            # NOTE: handling void labels isn't neat.
            panoptic_mask = pred["panoptic_seg"][0].cpu().detach().numpy()
            # handle void labels( -1 )
            panoptic_mask[panoptic_mask < 0] = 0
            parsed_pred["panoptic_mask"] = panoptic_mask

        return parsed_pred

    def _plot_table_row(self, pred):
        """
        plot prediction on one image

        Args:
            pred (Dict): Prediction for one image

        """
        file_name = pred["file_name"]
        classes = self._table_thing_classes
        # Process Bounding box detections
        boxes = {}
        avg_conf_per_class = [0 for i in range(len(self.thing_class_names))]
        counts = {}
        if pred.get("boxes") is not None:
            boxes_data = []
            for i, box in enumerate(pred["boxes"]):
                pred_class = int(pred["classes"][i])
                caption = (
                    f"{pred_class}"
                    if not self.thing_class_names
                    else self.thing_class_names[pred_class]
                )

                boxes_data.append(
                    {
                        "position": {
                            "minX": box[0],
                            "minY": box[1],
                            "maxX": box[2],
                            "maxY": box[3],
                        },
                        "class_id": pred_class + 1,
                        "box_caption": "%s %.3f" % (caption, pred["scores"][i]),
                        "scores": {"class_score": pred["scores"][i]},
                        "domain": "pixel",
                    }
                )

                avg_conf_per_class[pred_class] = avg_conf_per_class[pred_class] + pred["scores"][i]
                if pred_class in counts:
                    counts[pred_class] = counts[pred_class] + 1
                else:
                    counts[pred_class] = 1

            for pred_class in counts.keys():
                avg_conf_per_class[pred_class] = avg_conf_per_class[pred_class] / counts[pred_class]

            boxes = {
                "predictions": {
                    "box_data": boxes_data,
                    "class_labels": self.thing_index_to_class,
                }
            }

        masks = {}
        # pixles_per_class = [0 for _ in self.stuff_class_names]
        # Process semantic segmentation predictions
        if pred.get("sem_mask") is not None:
            masks["semantic_mask"] = {
                "mask_data": pred["sem_mask"],
                "class_labels": self.stuff_index_to_class,
            }
            classes = self._table_stuff_classes

        # TODO: Support panoptic , instance segmentation maks & keypoint visualizations.
        '''
        # Process instance segmentation detections
        if pred.get("pred_masks") is not None:
            class_count = {}
            num_pred = min(15, len(pred["pred_masks"]))  # Hardcoded to max 15 masks for better UI
            for i in range(num_pred):
                pred_class = int(pred["classes"][i])
                if pred_class in class_count:
                    class_count[pred_class] = class_count[pred_class] + 1
                else:
                    class_count[pred_class] = 0

                # title format - class_count. E.g - person_0, person_1 ..
                mask_title = (
                    f"class {pred_class}"
                    if not self.thing_class_names[loader_i]
                    else self.thing_class_names[loader_i][pred_class]
                )
                mask_title = f"{mask_title}_{class_count[pred_class]}"

                """
                masks[mask_title] = {
                    "mask_data": pred['pred_masks'][i]*(pred_class+1),
                    "class_labels": {pred_class+1: mask_title}
                }
                """

        if pred.get("panoptic_mask") is not None:
            masks["panoptic_mask"] = {
                "mask_data": pred["panoptic_mask"],
                "class_labels": self.stuff_index_to_class[loader_i],
            }
        '''

        table_row = [
            file_name,
            self.wandb.Image(file_name, boxes=boxes, masks=masks, classes=classes),
        ]
        if pred.get("boxes") is not None:
            table_row.extend(avg_conf_per_class)

        return table_row

    def _use_table_as_artifact(self, table):
        """
        This function logs the given table as artifact and calls `use_artifact`
        on it so tables from next iterations can use the reference of already uploaded images.

        Args:
            table (wandb.Table): Table to be logged
        """
        eval_art = self.wandb.Artifact(self._run.id + self.dataset_name, type="dataset")
        eval_art.add(table, self.dataset_name)
        self._run.use_artifact(eval_art)
        eval_art.wait()
        self._evalset_table_ref = eval_art.get(self.dataset_name).data

    def _build_evalset_table(self, pred_img):
        """
        Builds wandb.Table object for logging evaluation

        Args:
            pred_img (wandb.Image): An prediction in form of wandb Image

        returns:
            table (wandb.Table): Table object to log evaluation

        """
        # Current- Use cols. for each detection class score and don't use columns for overlays
        table_cols = ["file_name", "image"]
        if pred_img._boxes:
            table_cols = table_cols + self.thing_class_names
        table = self.wandb.Table(columns=table_cols)

        return table
