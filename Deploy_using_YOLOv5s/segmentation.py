import cv2
import time
import os
from ultralytics import YOLO
from datetime import datetime
import numpy as np


MODEL_PATH = os.getenv("PIGLET_MODEL_PATH", "best.pt")
ROOT_PATH = os.getenv("PIGLET_OUTPUT_DIR", "./temp_data")
CAMERA_USERNAME = os.getenv("PIGLET_CAMERA_USERNAME", "admin")
CAMERA_PASSWORD = os.getenv("PIGLET_CAMERA_PASSWORD")
CAMERA_IP_PREFIX = os.getenv("PIGLET_CAMERA_IP_PREFIX", "192.168.1")
CAMERA_PORT = int(os.getenv("PIGLET_CAMERA_PORT", "554"))
CAMERA_CHANNEL = int(os.getenv("PIGLET_CAMERA_CHANNEL", "101"))

if not CAMERA_PASSWORD:
    raise RuntimeError(
        "Missing PIGLET_CAMERA_PASSWORD. Set camera credentials with environment variables "
        "or a local .env file that is not committed."
    )

model = YOLO(MODEL_PATH)
root_path = ROOT_PATH
print("load model successfully")


def append_to_file(file_path, num_segments, average_area):
    with open(file_path, "a") as file:
        file.write(f"{num_segments}, {average_area}\n")


def calculate_area(mask):
    """Calculate the area of a binary mask."""
    return np.sum(mask)


def analyze_segmentation(results):
    """Analyze segmentation results and return instance count plus average area."""
    total_area = 0
    num_segments = 0
    for r in results:
        masks = r.masks
        if masks is not None:
            num_segments += len(masks.data)
            for mask in masks.data:
                binary_mask = mask.cpu().numpy().astype(bool)
                area = calculate_area(binary_mask)
                total_area += area
    if num_segments > 0:
        average_area = total_area / num_segments
    else:
        average_area = 0
    return num_segments, average_area


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")


# Logical pen IDs used for output file names.
id = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 54, 47, 48, 49, 50, 51, 52, 53, 46]

id_map = {
    28: 60, 29: 62, 31: 42, 32: 59, 33: 50, 34: 44, 35: 66, 36: 61, 37: 54,
    38: 49, 39: 55, 40: 57, 41: 56, 42: 63, 43: 53, 44: 41, 45: 48, 46: 64,
    47: 51, 48: 58, 49: 43, 50: 47, 51: 45, 52: 65, 53: 52, 54: 46
}

while True:
    current_time = datetime.now()
    current_date_str = current_time.strftime("%Y-%m-%d")
    temp_path = os.path.join(root_path, current_date_str)

    if 8 <= current_time.hour < 16:
        print("-------------------------")
        create_directory_if_not_exists(temp_path)

        for ch in id:
            # Map logical pen ID to the real camera IP suffix.
            ip_tail = id_map.get(ch, None)
            if ip_tail is None:
                print(f"{ch} has no mapping (likely removed). Skip.")
                continue

            camera_ip = f"{CAMERA_IP_PREFIX}.{ip_tail}"
            rtsp_url = (
                f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@"
                f"{camera_ip}:{CAMERA_PORT}/Streaming/Channels/{CAMERA_CHANNEL}"
            )

            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"{ch} Unable to connect to camera (real ip tail={ip_tail})")
                continue

            ret, frame = cap.read()
            if ret:
                results = model(frame)
                num_segments, average_area = analyze_segmentation(results)
                txt_filename = f"{ch}.txt"
                txt_path = os.path.join(temp_path, txt_filename)
                append_to_file(txt_path, num_segments, average_area)
            else:
                print(f"{ch} Unable to read frame (real ip tail={ip_tail})")

            cap.release()
    time.sleep(60 * 60)
