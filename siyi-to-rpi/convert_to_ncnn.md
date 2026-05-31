# Converting YOLOv5 → NCNN

Two options: download pre-converted weights, or export from scratch.

---

## Option A — Download pre-converted (fastest)

nihui maintains ready-to-use NCNN models:

    https://github.com/nihui/ncnn-assets/tree/master/models

Download these two files for yolov5s:
    yolov5s.ncnn.param
    yolov5s.ncnn.bin

Place them in:
    ./yolov5s_ncnn_model/

Then run:
    python3 yolo_rtsp_ncnn.py --model-dir ./yolov5s_ncnn_model --img-size 640

---

## Option B — Export from YOLOv5 repo (any model/size)

### Step 1 — On a PC/laptop (not RPi4)

    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt
    pip install onnx onnxruntime

    # Export to NCNN — choose your model and image size
    python export.py --weights yolov5s.pt --include ncnn --img 320
    # or
    python export.py --weights yolov5n.pt --include ncnn --img 320  # faster on RPi4

    # Output directory: yolov5s_ncnn_model/
    #   ├── yolov5s.ncnn.param
    #   └── yolov5s.ncnn.bin

### Step 2 — Copy to RPi4

    scp -r yolov5s_ncnn_model/ pi@<rpi4-ip>:~/detector/

### Step 3 — Run

    python3 yolo_rtsp_ncnn.py --model-dir ./yolov5s_ncnn_model --img-size 320

---

## Option C — Manual ONNX → NCNN (if export.py fails)

    # On PC: export to ONNX first
    python export.py --weights yolov5s.pt --include onnx --img 320 --simplify

    # Install NCNN tools
    # Ubuntu/Debian:
    sudo apt install ncnn-tools
    # or build from source: https://github.com/Tencent/ncnn

    # Convert
    onnx2ncnn yolov5s.onnx yolov5s.ncnn.param yolov5s.ncnn.bin

    # Optimise (optional but recommended)
    ncnnoptimize yolov5s.ncnn.param yolov5s.ncnn.bin \
                 yolov5s_opt.ncnn.param yolov5s_opt.ncnn.bin 0

---

## Model size vs RPi4 speed (320px input, 4 threads)

| Model     | Size   | Approx FPS |
|-----------|--------|------------|
| yolov5n   | ~4 MB  | ~12-15 fps |
| yolov5s   | ~14 MB | ~7-10 fps  |
| yolov5m   | ~42 MB | ~3-5 fps   |

Recommended: yolov5n or yolov5s at img-size 320.

---

## Install ncnn on RPi4

    pip install ncnn

If pip install fails (ARM build issues):

    sudo apt install cmake git
    git clone https://github.com/Tencent/ncnn
    cd ncnn
    git submodule update --init
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DNCNN_VULKAN=OFF \
          -DNCNN_BUILD_EXAMPLES=OFF \
          -DNCNN_PYTHON=ON ..
    make -j4
    cd ../python
    pip install .