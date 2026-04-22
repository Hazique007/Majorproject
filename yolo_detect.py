import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Safe picamera2 import
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# ── Argument parsing ──────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model',      required=True,
                    help='Path to YOLO model (e.g. my_model.pt)')
parser.add_argument('--source',     required=True,
                    help='Source: image.jpg / folder / video.mp4 / usb0 / picamera0')
parser.add_argument('--thresh',     default=0.5,  type=float,
                    help='Confidence threshold (default 0.5)')
parser.add_argument('--resolution', default=None,
                    help='Display resolution WxH (e.g. 640x480)')
parser.add_argument('--record',     action='store_true',
                    help='Record output as demo1.avi (requires --resolution)')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res   = args.resolution
record     = args.record

# ── Validate model ────────────────────────────
if not os.path.exists(model_path):
    print(f'ERROR: Model not found at "{model_path}"')
    sys.exit(0)

model  = YOLO(model_path, task='detect')
labels = model.names

# ── Determine source type ─────────────────────
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'ERROR: File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx     = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx   = int(img_source[8:])
elif 'youtube.com' in img_source or 'youtu.be' in img_source:
    import yt_dlp
    with yt_dlp.YoutubeDL({'format': 'best'}) as ydl:
        info       = ydl.extract_info(img_source, download=False)
        img_source = info['url']
    source_type = 'video'
else:
    print(f'ERROR: Source "{img_source}" is invalid.')
    sys.exit(0)

# ── Parse resolution ──────────────────────────
resize = False
if user_res:
    resize = True
    resW   = int(user_res.split('x')[0])
    resH   = int(user_res.split('x')[1])

# ── Recording setup ───────────────────────────
if record:
    if source_type not in ['video', 'usb']:
        print('ERROR: Recording only works for video and camera sources.')
        sys.exit(0)
    if not user_res:
        print('ERROR: --resolution is required when using --record')
        sys.exit(0)
    recorder = cv2.VideoWriter(
        'demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH)
    )

# ── Initialize source ─────────────────────────
if source_type == 'image':
    imgs_list = [img_source]

elif source_type == 'folder':
    imgs_list = [
        f for f in glob.glob(img_source + '/*')
        if os.path.splitext(f)[1] in img_ext_list
    ]

elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap     = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

elif source_type == 'picamera':
    if not PICAMERA_AVAILABLE:
        print('ERROR: picamera2 is not installed.')
        print('Fix:   sudo apt install python3-picamera2')
        sys.exit(0)
    if not user_res:
        print('ERROR: --resolution is required for picamera (e.g. --resolution 640x480)')
        sys.exit(0)
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(
        main={"format": 'XRGB8888', "size": (resW, resH)}
    ))
    cap.start()

# ── Colours ───────────────────────────────────
bbox_colors = [
    (164,120,87),(68,148,228),(93,97,209),(178,182,133),(88,159,106),
    (96,202,231),(159,124,168),(169,162,241),(98,118,150),(172,176,184)
]

# ── Inference loop ────────────────────────────
avg_frame_rate   = 0
frame_rate_buffer = []
fps_avg_len      = 200
img_count        = 0

while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed. Exiting.')
            sys.exit(0)
        frame     = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('End of video. Exiting.')
            break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if frame is None or not ret:
            print('Camera disconnected. Exiting.')
            break

    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame      = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if frame is None:
            print('Picamera read failed. Exiting.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run YOLO
    results    = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):
        xyxy   = detections[i].xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx  = int(detections[i].cls.item())
        classname = labels[classidx]
        conf      = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label     = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame,
                (xmin, label_ymin - labelSize[1] - 10),
                (xmin + labelSize[0], label_ymin + baseLine - 10),
                color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    # HUD
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}',
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.putText(frame, f'Faces: {object_count}',
        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.imshow('YOLO Face Detection', frame)
    if record:
        recorder.write(frame)

    # Key handling
    key = cv2.waitKey() if source_type in ['image','folder'] else cv2.waitKey(5)
    if   key in [ord('q'), ord('Q')]: break
    elif key in [ord('s'), ord('S')]: cv2.waitKey()
    elif key in [ord('p'), ord('P')]: cv2.imwrite('capture.png', frame)

    # FPS calc
    t_stop           = time.perf_counter()
    frame_rate_calc  = 1.0 / (t_stop - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# ── Cleanup ───────────────────────────────────
print(f'Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()