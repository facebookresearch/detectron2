#!/usr/bin/env python
import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.utils.visualizer import Visualizer


LOGGER = logging.getLogger("oweed_full_val_infer_annotate")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--dummy-size", type=int, default=1024)
    parser.add_argument(
        "--postprocess-mode",
        choices=("full-mask", "lowres-contour"),
        default="full-mask",
        help=(
            "full-mask uses Detectron2's standard postprocess and materializes full-resolution "
            "masks. lowres-contour keeps ROI masks compact, extracts contours at model input "
            "resolution, and scales contour points to the original image."
        ),
    )
    parser.add_argument(
        "--contours-json",
        default=None,
        help="Output JSON path for lowres-contour polygon predictions. Defaults inside output-dir.",
    )
    parser.add_argument("--contour-alpha", type=float, default=0.45)
    return parser.parse_args()


def gpu_mem():
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


def image_files(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)


def run_model(model, image_rgb, resize_aug, do_postprocess=True):
    original_h, original_w = image_rgb.shape[:2]
    transform = resize_aug.get_transform(image_rgb)
    resized = transform.apply_image(image_rgb)
    tensor = torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1)))
    inputs = {
        "image": tensor,
        "height": original_h,
        "width": original_w,
    }
    if do_postprocess:
        outputs = model([inputs])[0]["instances"]
    else:
        outputs = model.inference([inputs], do_postprocess=False)[0]
    return outputs, resized.shape[:2]


def timed_inference(model, image_rgb, resize_aug, do_postprocess=True):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    outputs, resized_hw = run_model(model, image_rgb, resize_aug, do_postprocess=do_postprocess)
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    gpu_ms = start_event.elapsed_time(end_event)
    return outputs, resized_hw, gpu_ms, wall_seconds


def draw_predictions(image_rgb, outputs, metadata, output_path):
    vis = Visualizer(image_rgb, metadata=metadata)
    drawn = vis.draw_instance_predictions(outputs.to("cpu"))
    out_rgb = drawn.get_image()
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), out_bgr)


def color_for_class(class_id):
    # Deterministic bright-ish BGR color without depending on Visualizer internals.
    palette = np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
            [128, 128, 0],
            [255, 215, 180],
            [0, 0, 128],
            [128, 128, 128],
        ],
        dtype=np.uint8,
    )
    rgb = palette[class_id % len(palette)]
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def mask_contours_from_raw_instance(mask_28, box_xyxy, resized_hw, original_hw, mask_thresh):
    resized_h, resized_w = resized_hw
    original_h, original_w = original_hw

    x0, y0, x1, y1 = box_xyxy.astype(float).tolist()
    x0_i = max(0, min(resized_w - 1, int(np.floor(x0))))
    y0_i = max(0, min(resized_h - 1, int(np.floor(y0))))
    x1_i = max(0, min(resized_w, int(np.ceil(x1))))
    y1_i = max(0, min(resized_h, int(np.ceil(y1))))

    crop_w = x1_i - x0_i
    crop_h = y1_i - y0_i
    if crop_w <= 1 or crop_h <= 1:
        return []

    resized_mask = cv2.resize(mask_28, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    binary = (resized_mask >= mask_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale_x = original_w / float(resized_w)
    scale_y = original_h / float(resized_h)
    scaled_contours = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        pts = contour.reshape(-1, 2).astype(np.float32)
        pts[:, 0] = (pts[:, 0] + x0_i) * scale_x
        pts[:, 1] = (pts[:, 1] + y0_i) * scale_y
        pts[:, 0] = np.clip(pts[:, 0], 0, original_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, original_h - 1)
        scaled_contours.append(np.rint(pts).astype(np.int32))
    return scaled_contours


def raw_outputs_to_contours(outputs, resized_hw, original_hw, metadata, mask_thresh):
    outputs_cpu = outputs.to("cpu")
    boxes_resized = outputs_cpu.pred_boxes.tensor.numpy()
    scores = outputs_cpu.scores.numpy()
    classes = outputs_cpu.pred_classes.numpy()
    masks = outputs_cpu.pred_masks[:, 0].numpy()

    resized_h, resized_w = resized_hw
    original_h, original_w = original_hw
    scale_x = original_w / float(resized_w)
    scale_y = original_h / float(resized_h)
    class_names = getattr(metadata, "thing_classes", None)

    predictions = []
    for idx, (box, score, class_id, mask) in enumerate(zip(boxes_resized, scores, classes, masks)):
        contours = mask_contours_from_raw_instance(
            mask, box, resized_hw, original_hw, mask_thresh=mask_thresh
        )
        if not contours:
            continue

        box_full = [
            float(np.clip(box[0] * scale_x, 0, original_w - 1)),
            float(np.clip(box[1] * scale_y, 0, original_h - 1)),
            float(np.clip(box[2] * scale_x, 0, original_w - 1)),
            float(np.clip(box[3] * scale_y, 0, original_h - 1)),
        ]
        predictions.append(
            {
                "instance_index": idx,
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)] if class_names else str(int(class_id)),
                "score": float(score),
                "bbox_xyxy_resized": [float(v) for v in box.tolist()],
                "bbox_xyxy_fullres": box_full,
                "contours_fullres": [c.reshape(-1, 2).astype(int).tolist() for c in contours],
            }
        )
    return predictions


def draw_contour_predictions(image_rgb, contour_predictions, output_path, alpha):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = image_bgr.copy()

    for pred in contour_predictions:
        color = color_for_class(pred["class_id"])
        contours = [np.asarray(c, dtype=np.int32).reshape(-1, 1, 2) for c in pred["contours_fullres"]]
        if not contours:
            continue
        cv2.fillPoly(overlay, contours, color)
        cv2.polylines(image_bgr, contours, isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    image_bgr = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)

    for pred in contour_predictions:
        color = color_for_class(pred["class_id"])
        x0, y0, x1, y1 = [int(round(v)) for v in pred["bbox_xyxy_fullres"]]
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)
        label = f'{pred["class_name"]} {pred["score"]:.2f}'
        label_y = max(20, y0 - 5)
        cv2.putText(
            image_bgr,
            label,
            (x0, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image_bgr)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = LazyConfig.load(args.config)
    cfg.model._target_ = GeneralizedRCNN
    cfg.model.roi_heads.box_predictor.test_score_thresh = args.score_thresh
    cfg.model.roi_heads.box_predictor.test_nms_thresh = args.nms_thresh
    cfg.train.init_checkpoint = args.checkpoint

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model.eval()
    DetectionCheckpointer(model).load(args.checkpoint)

    dataset_name = cfg.dataloader.evaluator.dataset_name
    metadata = MetadataCatalog.get(dataset_name)
    resize_aug = ResizeShortestEdge(short_edge_length=1024, max_size=1024)

    files = image_files(args.image_dir)
    if not files:
        raise RuntimeError(f"No images found in {args.image_dir}")

    run_log = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "image_dir": args.image_dir,
        "output_dir": str(out_dir),
        "score_thresh": args.score_thresh,
        "nms_thresh": args.nms_thresh,
        "mask_thresh": args.mask_thresh,
        "postprocess_mode": args.postprocess_mode,
        "resize_short_edge": 1024,
        "resize_max_size": 1024,
        "num_images": len(files),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warmup": {},
        "images": [],
    }
    contour_records = {
        "format": "oweed_scaled_lowres_contours_v1",
        "description": (
            "Contours are extracted from compact ROI masks at resized model-input scale, "
            "then scaled to original image coordinates. Dense full-resolution masks are not stored."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "image_dir": args.image_dir,
        "score_thresh": args.score_thresh,
        "nms_thresh": args.nms_thresh,
        "mask_thresh": args.mask_thresh,
        "resize_short_edge": 1024,
        "resize_max_size": 1024,
        "images": [],
    }
    contours_json_path = (
        Path(args.contours_json)
        if args.contours_json
        else out_dir / "contours_fullres_scaled_from_lowres.json"
    )
    do_postprocess = args.postprocess_mode == "full-mask"

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        dummy = torch.zeros(
            3, args.dummy_size, args.dummy_size, dtype=torch.uint8, device="cpu"
        ).numpy()
        dummy_rgb = np.transpose(dummy, (1, 2, 0))
        warmup_times = []
        for _ in range(args.warmup_iters):
            _, _, gpu_ms, wall_seconds = timed_inference(
                model, dummy_rgb, resize_aug, do_postprocess=do_postprocess
            )
            warmup_times.append({"gpu_ms": gpu_ms, "wall_seconds": wall_seconds})
        run_log["warmup"] = {
            "iters": args.warmup_iters,
            "dummy_hwc": [args.dummy_size, args.dummy_size, 3],
            "times": warmup_times,
            "memory_after_warmup": gpu_mem(),
        }

        torch.cuda.reset_peak_memory_stats()
        total_gpu_ms = 0.0
        total_wall_seconds = 0.0
        total_contour_seconds = 0.0
        total_draw_seconds = 0.0
        max_detections = 0
        for index, path in enumerate(files, start=1):
            image_rgb = read_image(str(path), format="RGB")
            torch.cuda.reset_peak_memory_stats()
            outputs, resized_hw, gpu_ms, wall_seconds = timed_inference(
                model, image_rgb, resize_aug, do_postprocess=do_postprocess
            )
            num_instances = len(outputs)
            max_detections = max(max_detections, num_instances)
            total_gpu_ms += gpu_ms
            total_wall_seconds += wall_seconds

            output_path = (
                out_dir / f"{path.stem}_pred_conf{args.score_thresh:.2f}_nms{args.nms_thresh:.2f}.jpg"
            )
            contour_seconds = 0.0
            draw_seconds = 0.0
            kept_contours = None
            if args.postprocess_mode == "full-mask":
                draw_start = time.perf_counter()
                draw_predictions(image_rgb, outputs, metadata, output_path)
                draw_seconds = time.perf_counter() - draw_start
            else:
                contour_start = time.perf_counter()
                contour_predictions = raw_outputs_to_contours(
                    outputs,
                    resized_hw,
                    image_rgb.shape[:2],
                    metadata,
                    mask_thresh=args.mask_thresh,
                )
                contour_seconds = time.perf_counter() - contour_start
                kept_contours = len(contour_predictions)

                draw_start = time.perf_counter()
                draw_contour_predictions(
                    image_rgb,
                    contour_predictions,
                    output_path,
                    alpha=args.contour_alpha,
                )
                draw_seconds = time.perf_counter() - draw_start

                contour_records["images"].append(
                    {
                        "file": str(path),
                        "output": str(output_path),
                        "original_hw": list(image_rgb.shape[:2]),
                        "resized_hw": list(resized_hw),
                        "num_raw_detections": num_instances,
                        "num_contour_predictions": kept_contours,
                        "predictions": contour_predictions,
                    }
                )

            total_contour_seconds += contour_seconds
            total_draw_seconds += draw_seconds

            record = {
                "index": index,
                "file": str(path),
                "output": str(output_path),
                "original_hw": list(image_rgb.shape[:2]),
                "resized_hw": list(resized_hw),
                "detections": num_instances,
                "contour_predictions": kept_contours,
                "gpu_ms": gpu_ms,
                "wall_seconds": wall_seconds,
                "contour_seconds": contour_seconds,
                "draw_seconds": draw_seconds,
                "memory": gpu_mem(),
            }
            run_log["images"].append(record)
            LOGGER.info(
                "%03d/%03d %s det=%d contours=%s gpu_ms=%.2f wall=%.3fs peak_alloc=%.1fMB",
                index,
                len(files),
                path.name,
                num_instances,
                "-" if kept_contours is None else kept_contours,
                gpu_ms,
                wall_seconds,
                record["memory"]["max_allocated_mb"],
            )

    run_log["summary"] = {
        "total_gpu_seconds": total_gpu_ms / 1000.0,
        "total_wall_seconds_model_only": total_wall_seconds,
        "avg_gpu_ms_per_image": total_gpu_ms / len(files),
        "avg_wall_seconds_model_only": total_wall_seconds / len(files),
        "total_contour_seconds": total_contour_seconds,
        "avg_contour_seconds": total_contour_seconds / len(files),
        "total_draw_seconds": total_draw_seconds,
        "avg_draw_seconds": total_draw_seconds / len(files),
        "max_detections_per_image": max_detections,
        "final_memory": gpu_mem(),
    }
    log_path = out_dir / "inference_timing_gpu_mem.json"
    log_path.write_text(json.dumps(run_log, indent=2))
    if args.postprocess_mode == "lowres-contour":
        contours_json_path.write_text(json.dumps(contour_records, indent=2))
        LOGGER.info("Wrote scaled contour predictions to %s", contours_json_path)
    LOGGER.info("Wrote timing/memory log to %s", log_path)


if __name__ == "__main__":
    main()
