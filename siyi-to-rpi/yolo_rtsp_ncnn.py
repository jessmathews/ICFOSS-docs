#!/usr/bin/env python3
"""
YOLOv5 NCNN Object Detector — Raspberry Pi 4
Uses ncnn (ARM-optimised inference) instead of PyTorch for 3-5x faster speed.

RTSP stream: rtsp://192.168.144.25:8554/video1

Setup:
    pip install ncnn opencv-python numpy

Model files (place in same directory or use --model-dir):
    yolov5s_ncnn_model/
        ├── yolov5s.ncnn.param
        └── yolov5s.ncnn.bin

    Download pre-converted models:
        https://github.com/nihui/ncnn-assets/tree/master/models
    OR convert yourself — see convert_to_ncnn.md

Usage:
    python3 yolo_rtsp_ncnn.py                   # default detection
    python3 yolo_rtsp_ncnn.py --debug            # stream diagnostics only
    python3 yolo_rtsp_ncnn.py --show             # display window
    python3 yolo_rtsp_ncnn.py --model-dir ./models/yolov5n_ncnn_model
    python3 yolo_rtsp_ncnn.py --conf 0.4 --threads 4
"""

import argparse
import logging
import os
import sys
import time
import socket
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# COCO class names
# ---------------------------------------------------------------------------
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# Deterministic per-class colours
CLASS_COLORS = [
    tuple(int(c) for c in (
        (i * 67 + 80) % 255,
        (i * 113 + 140) % 255,
        (i * 41 + 200) % 255
    )) for i in range(len(COCO_CLASSES))
]

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv5 NCNN Object Detector — Raspberry Pi 4"
    )
    parser.add_argument("--rtsp", default="rtsp://192.168.144.25:8554/video1")
    parser.add_argument(
        "--model-dir", default="./yolov5s_ncnn_model",
        help="Directory containing .param and .bin files"
    )
    parser.add_argument(
        "--param", default=None,
        help="Explicit path to .param file (overrides --model-dir)"
    )
    parser.add_argument(
        "--bin", default=None,
        help="Explicit path to .bin file (overrides --model-dir)"
    )
    parser.add_argument("--conf",    type=float, default=0.35)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--img-size",type=int,   default=320,
                        help="Input size (must match model — typically 320 or 640)")
    parser.add_argument("--threads", type=int,   default=4,
                        help="NCNN thread count (RPi4 has 4 cores, default=4)")
    parser.add_argument("--classes", nargs="+",  type=int, default=None,
                        help="Filter class indices e.g. --classes 0 2")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Run inference every N frames (default=1 with ncnn)")
    parser.add_argument("--show",   action="store_true")
    parser.add_argument("--save",   action="store_true")
    parser.add_argument("--output", default="./detections")
    parser.add_argument("--debug",  action="store_true",
                        help="Stream diagnostics mode (no inference)")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(debug, log_file=None):
    level = logging.DEBUG if debug else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers
    )
    return logging.getLogger("yolo_ncnn")


# ---------------------------------------------------------------------------
# Debug / stream check
# ---------------------------------------------------------------------------

def debug_stream(rtsp_url, logger):
    from urllib.parse import urlparse

    logger.info("=" * 60)
    logger.info("  DEBUG MODE — RTSP Stream Diagnostics")
    logger.info("=" * 60)
    logger.info(f"  URL: {rtsp_url}\n")

    parsed = urlparse(rtsp_url)
    host   = parsed.hostname
    port   = parsed.port or 554

    # 1. TCP
    logger.info(f"[1/5] TCP socket → {host}:{port}")
    try:
        s = socket.create_connection((host, port), timeout=5)
        s.close()
        logger.info("      ✓ TCP OK")
    except Exception as e:
        logger.error(f"      ✗ TCP failed: {e}")

    # 2. Ping
    logger.info(f"[2/5] Ping → {host}")
    try:
        r = subprocess.run(["ping","-c","3","-W","2", host],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            lat = [l for l in r.stdout.splitlines() if "avg" in l]
            logger.info(f"      ✓ {lat[-1] if lat else 'reachable'}")
        else:
            logger.warning("      ✗ Ping failed (ICMP may be blocked)")
    except Exception as e:
        logger.warning(f"      ✗ {e}")

    # 3. ncnn import
    logger.info("[3/5] ncnn import check")
    try:
        import ncnn
        logger.info(f"      ✓ ncnn {ncnn.__version__}")
        logger.info(f"        vulkan={ncnn.get_gpu_count() > 0}  "
                    f"threads_supported=True")
    except ImportError:
        logger.error("      ✗ ncnn not installed → pip install ncnn")

    # 4. OpenCV / stream open
    logger.info(f"[4/5] OpenCV stream open")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("      ✗ Could not open stream")
        logger.warning("        → ffplay -rtsp_transport tcp \"" + rtsp_url + "\"")
        return False
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fcc >> (8*i)) & 0xFF) for i in range(4)])
    logger.info(f"      ✓ Opened  {w}×{h} @ {fps:.1f}fps  codec={codec}")

    # 5. Frame grab
    logger.info("[5/5] Frame grab test (10 frames)")
    ok = 0
    times = []
    for _ in range(10):
        t0 = time.time()
        ret, frame = cap.read()
        dt = time.time() - t0
        if ret and frame is not None:
            ok += 1
            times.append(dt)
    cap.release()
    avg_ms = sum(times)/len(times)*1000 if times else 0
    logger.info(f"      ✓ {ok}/10 frames  avg_decode={avg_ms:.1f}ms")

    logger.info("")
    logger.info("=" * 60)
    if ok >= 5:
        logger.info("  ✓ Stream check PASSED")
        logger.info("  Run without --debug to start NCNN inference")
    else:
        logger.info("  ✗ Stream check FAILED — low frame count")
    logger.info("=" * 60)
    return ok >= 5


# ---------------------------------------------------------------------------
# NCNN YOLOv5 wrapper
# ---------------------------------------------------------------------------

class YOLOv5NCNN:
    """
    Thin wrapper around ncnn Net for YOLOv5 inference.

    Handles:
      - letterbox resize
      - forward pass through ncnn
      - decode YOLOv5 output blobs (3 scales)
      - NMS
    """

    # YOLOv5 anchor grid (default — matches official export)
    ANCHORS = [
        [10,13, 16,30, 33,23],    # P3/8  small
        [30,61, 62,45, 59,119],   # P4/16 medium
        [116,90, 156,198, 373,326] # P5/32 large
    ]
    STRIDES = [8, 16, 32]

    def __init__(self, param_path, bin_path, img_size=320,
                 conf=0.35, iou=0.45, num_threads=4, logger=None):
        import ncnn

        self.img_size   = img_size
        self.conf       = conf
        self.iou        = iou
        self.logger     = logger or logging.getLogger(__name__)
        self.num_classes = len(COCO_CLASSES)

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False   # CPU only on RPi4
        self.net.opt.num_threads        = num_threads

        ret_p = self.net.load_param(str(param_path))
        ret_b = self.net.load_model(str(bin_path))

        if ret_p != 0 or ret_b != 0:
            raise RuntimeError(
                f"Failed to load NCNN model  param={ret_p}  bin={ret_b}\n"
                f"  param: {param_path}\n"
                f"  bin  : {bin_path}"
            )

        self.logger.info(f"✓ NCNN model loaded  threads={num_threads}  "
                         f"img_size={img_size}")

    # ------------------------------------------------------------------
    def _letterbox(self, img):
        """Resize keeping aspect ratio, pad to square."""
        h, w = img.shape[:2]
        s = self.img_size
        scale = min(s / w, s / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        pad_x = (s - nw) // 2
        pad_y = (s - nh) // 2
        canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
        return canvas, scale, pad_x, pad_y

    # ------------------------------------------------------------------
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------
    def _decode_blob(self, blob, stride, anchors):
        """Decode a single YOLOv5 output blob."""
        import ncnn
        # blob shape: (num_anchors, grid_h, grid_w, 5 + num_classes)
        # ncnn returns as flat Mat — reshape
        na = 3  # anchors per scale
        nc = self.num_classes
        s  = self.img_size // stride
        data = np.array(blob)  # flat

        # Depending on ncnn model export, shape may vary.
        # Typical yolov5 ncnn export: output blob is (1, na*(5+nc), grid, grid)
        # Already decoded by yolov5's detect layer in some exports.
        # We handle both cases below.
        try:
            data = data.reshape(na, s, s, 5 + nc)
        except ValueError:
            # Some exports flatten differently
            data = data.reshape(1, na * (5 + nc), s, s)
            data = data[0].reshape(na, 5 + nc, s, s).transpose(0, 2, 3, 1)

        boxes, scores, class_ids = [], [], []
        ax = np.arange(s)
        grid_y, grid_x = np.meshgrid(ax, ax, indexing="ij")  # (s,s)

        for a_idx in range(na):
            pred = data[a_idx]  # (s, s, 5+nc)
            pred = self._sigmoid(pred)

            obj_conf = pred[..., 4]                # (s,s)
            cls_conf = pred[..., 5:] * obj_conf[..., None]  # (s,s,nc)

            mask = obj_conf > self.conf
            if not np.any(mask):
                continue

            cx = (pred[..., 0] * 2 - 0.5 + grid_x) * stride
            cy = (pred[..., 1] * 2 - 0.5 + grid_y) * stride
            bw = (pred[..., 2] * 2) ** 2 * anchors[a_idx * 2]
            bh = (pred[..., 3] * 2) ** 2 * anchors[a_idx * 2 + 1]

            for gy in range(s):
                for gx in range(s):
                    if not mask[gy, gx]:
                        continue
                    cls_id = int(np.argmax(cls_conf[gy, gx]))
                    score  = float(cls_conf[gy, gx, cls_id])
                    if score < self.conf:
                        continue
                    x1 = float(cx[gy, gx] - bw[gy, gx] / 2)
                    y1 = float(cy[gy, gx] - bh[gy, gx] / 2)
                    x2 = float(cx[gy, gx] + bw[gy, gx] / 2)
                    y2 = float(cy[gy, gx] + bh[gy, gx] / 2)
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    class_ids.append(cls_id)

        return boxes, scores, class_ids

    # ------------------------------------------------------------------
    def infer(self, frame_bgr):
        """
        Run inference on a BGR frame.
        Returns list of dicts: {x1,y1,x2,y2, conf, class_id, label}
        """
        import ncnn

        orig_h, orig_w = frame_bgr.shape[:2]
        img_lb, scale, pad_x, pad_y = self._letterbox(frame_bgr)
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

        mat_in = ncnn.Mat.from_pixels(
            img_rgb,
            ncnn.Mat.PixelType.PIXEL_RGB,
            self.img_size, self.img_size
        )
        # Normalise 0-255 → 0-1
        mean_vals = [0.0, 0.0, 0.0]
        norm_vals = [1/255.0, 1/255.0, 1/255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)

        ex = self.net.create_extractor()
        ex.set_num_threads(self.net.opt.num_threads)

        # Input layer name for yolov5 ncnn export is typically "images"
        ex.input("images", mat_in)

        all_boxes, all_scores, all_cls = [], [], []

        # Output layers: output, 331, 329  (yolov5s default ncnn export)
        # OR: output0 / stride_8 / stride_16 / stride_32
        # We try both naming conventions
        output_names_options = [
            ["output", "331", "329"],          # yolov5 ncnn-assets naming
            ["output0", "output1", "output2"], # some converters
            ["stride_8", "stride_16", "stride_32"],
        ]

        extracted = False
        for output_names in output_names_options:
            try:
                blobs = []
                for name in output_names:
                    ret, blob = ex.extract(name)
                    if ret != 0:
                        break
                    blobs.append(blob)
                if len(blobs) == 3:
                    for i, (blob, stride, anc) in enumerate(
                        zip(blobs, self.STRIDES, self.ANCHORS)
                    ):
                        b, s, c = self._decode_blob(blob, stride, anc)
                        all_boxes.extend(b)
                        all_scores.extend(s)
                        all_cls.extend(c)
                    extracted = True
                    break
            except Exception:
                continue

        if not extracted:
            self.logger.warning(
                "Could not extract output blobs — check output layer names. "
                "Run with --debug and inspect your .param file."
            )
            return []

        if not all_boxes:
            return []

        # NMS
        boxes_xywh = [
            [b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in all_boxes
        ]
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, all_scores, self.conf, self.iou
        )
        if len(indices) == 0:
            return []

        results = []
        for idx in indices.flatten():
            x1, y1, x2, y2 = all_boxes[idx]
            # Unpad + unscale back to original frame coords
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            x1 = max(0, min(int(x1), orig_w))
            y1 = max(0, min(int(y1), orig_h))
            x2 = max(0, min(int(x2), orig_w))
            y2 = max(0, min(int(y2), orig_h))
            cls_id = all_cls[idx]
            results.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": float(all_scores[idx]),
                "class_id": cls_id,
                "label": COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
            })
        return results


# ---------------------------------------------------------------------------
# Draw detections
# ---------------------------------------------------------------------------

def draw_detections(frame, detections, filter_classes=None):
    for det in detections:
        cls_id = det["class_id"]
        if filter_classes and cls_id not in filter_classes:
            continue
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label  = det["label"]
        conf   = det["conf"]
        colour = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), colour, -1)
        cv2.putText(frame, text, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


# ---------------------------------------------------------------------------
# Resolve model paths
# ---------------------------------------------------------------------------

def resolve_model_paths(args, logger):
    if args.param and args.bin:
        return Path(args.param), Path(args.bin)

    model_dir = Path(args.model_dir)
    params = list(model_dir.glob("*.param"))
    bins   = list(model_dir.glob("*.bin"))

    if not params or not bins:
        logger.error(f"No .param or .bin files found in: {model_dir}")
        logger.error("")
        logger.error("Download pre-converted YOLOv5 NCNN models:")
        logger.error("  https://github.com/nihui/ncnn-assets/tree/master/models")
        logger.error("")
        logger.error("Or convert from PyTorch:")
        logger.error("  git clone https://github.com/ultralytics/yolov5")
        logger.error("  cd yolov5")
        logger.error("  pip install onnx onnxruntime")
        logger.error("  python export.py --weights yolov5s.pt --include ncnn "
                     "--img 320")
        logger.error("  # Output: yolov5s_ncnn_model/")
        sys.exit(1)

    if len(params) > 1:
        logger.warning(f"Multiple .param files found, using: {params[0]}")
    if len(bins) > 1:
        logger.warning(f"Multiple .bin files found, using: {bins[0]}")

    return params[0], bins[0]


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def run_detection(args, logger):
    param_path, bin_path = resolve_model_paths(args, logger)
    logger.info(f"Model param : {param_path}")
    logger.info(f"Model bin   : {bin_path}")

    try:
        detector = YOLOv5NCNN(
            param_path, bin_path,
            img_size=args.img_size,
            conf=args.conf,
            iou=args.iou,
            num_threads=args.threads,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        sys.exit(1)

    if args.save:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving frames to: {args.output}")

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    logger.info(f"Connecting to: {args.rtsp}")
    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        logger.error("Cannot open RTSP stream. Run with --debug to diagnose.")
        sys.exit(1)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info(f"Stream: {w}×{h} @ {fps:.1f}fps")
    logger.info("Press Ctrl+C to stop\n")

    frame_idx    = 0
    last_dets    = []
    fps_counter  = 0
    fps_t0       = time.time()
    inf_times    = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Frame read failed — reconnecting in 2s ...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
                continue

            frame_idx += 1

            if frame_idx % args.skip_frames == 0:
                t0 = time.time()
                last_dets = detector.infer(frame)
                inf_ms = (time.time() - t0) * 1000
                inf_times.append(inf_ms)
                logger.debug(
                    f"Frame {frame_idx:05d}  inf={inf_ms:.1f}ms  "
                    f"det={len(last_dets)}"
                )

            annotated = frame.copy()
            draw_detections(annotated, last_dets, args.classes)

            fps_counter += 1
            if time.time() - fps_t0 >= 2.0:
                disp_fps = fps_counter / (time.time() - fps_t0)
                avg_inf  = sum(inf_times[-30:]) / max(len(inf_times[-30:]), 1)
                fps_counter = 0
                fps_t0 = time.time()
                logger.info(
                    f"FPS:{disp_fps:.1f}  Inf:{avg_inf:.0f}ms  "
                    f"Det:{len(last_dets)}"
                )

            # Overlay HUD
            cv2.putText(
                annotated,
                f"NCNN YOLOv5 | det:{len(last_dets)} | "
                f"{'%.0f'%inf_times[-1]}ms" if inf_times else "NCNN YOLOv5",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1
            )

            if args.show:
                cv2.imshow("YOLOv5 NCNN — RTSP", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if args.save and frame_idx % (args.skip_frames * 5) == 0:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(os.path.join(args.output, f"frame_{ts}.jpg"), annotated)

    except KeyboardInterrupt:
        logger.info("\nStopped by user.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if inf_times:
            logger.info(f"Avg inference : {sum(inf_times)/len(inf_times):.1f}ms")
            logger.info(f"Frames read   : {frame_idx}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    logger = setup_logging(args.debug, args.log_file)

    logger.info("YOLOv5 NCNN Detector — Raspberry Pi 4")
    logger.info(f"Stream    : {args.rtsp}")
    logger.info(f"Model dir : {args.model_dir}")
    logger.info(f"Img size  : {args.img_size}  Conf:{args.conf}  IOU:{args.iou}")
    logger.info(f"Threads   : {args.threads}  Skip:{args.skip_frames}\n")

    if args.debug:
        ok = debug_stream(args.rtsp, logger)
        sys.exit(0 if ok else 1)
    else:
        run_detection(args, logger)


if __name__ == "__main__":
    main()