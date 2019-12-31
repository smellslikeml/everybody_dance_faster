# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import time
import numpy as np
from PIL import Image
import svgwrite
from cairosvg import svg2png
import cv2

from pose_engine import PoseEngine
from google.cloud import storage

EDGES = (
    ('nose', 'left eye', 'rgb(230,25,75)'),
    ('nose', 'right eye', 'rgb(60,180,75)'),
    ('nose', 'left ear', 'rgb(255,225,25)'),
    ('nose', 'right ear', 'rgb(0,130,200)'),
    ('left ear', 'left eye', 'rgb(245,130,48)'),
    ('right ear', 'right eye','rgb(145,30,180)'),
    ('left eye', 'right eye', 'rgb(70,240,240)'),
    ('left shoulder', 'right shoulder', 'rgb(240,50,230)'),
    ('left shoulder', 'left elbow', 'rgb(210,245,60)'),
    ('left shoulder', 'left hip', 'rgb(250,190,190)'),
    ('right shoulder', 'right elbow', 'rgb(0,128,128)'),
    ('right shoulder', 'right hip', 'rgb(230,190,255)'),
    ('left elbow', 'left wrist', 'rgb(170,110,40)'),
    ('right elbow', 'right wrist','rgb(255,250,200)'),
    ('left hip', 'right hip', 'rgb(128,0,0)'),
    ('left hip', 'left knee', 'rgb(170,255,195)'),
    ('right hip', 'right knee', 'rgb(128, 128, 0)'),
    ('left knee', 'left ankle', 'rgb(255,215,180)'),
    ('right knee', 'right ankle', 'rgb(0,0,128)'),
)


def draw_pose(dwg, pose, src_size, inference_box, color='cyan', threshold=0.2):
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        # Offset and scale to source coordinate space.
        kp_y = int((keypoint.yx[0] - box_y) * scale_y)
        kp_x = int((keypoint.yx[1] - box_x) * scale_x)

        xys[label] = (kp_x, kp_y)
        dwg.add(dwg.circle(center=(int(kp_x), int(kp_y)), r=5,
                           fill='cyan', fill_opacity=keypoint.score, stroke=color))

    for a, b, c in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        dwg.add(dwg.line(start=(ax, ay), end=(bx, by), stroke=c, stroke_width=2))

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_file(source_file_name)

    print(
        "File uploaded to {}.".format(
            destination_blob_name
        )
    )

model = 'models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'
print('Loading model: ', model)
engine = PoseEngine(model)
input_shape = engine.get_input_tensor_shape()
inference_size = (input_shape[2], input_shape[1])
src_size = (512, 512)
bucket_name = 'YOUR-BUCKET-HERE'

cap = cv2.VideoCapture(0)

def main():
    while True:
        ret, frame = cap.read()
	pil_image = Image.fromarray(frame)
	pil_image.resize((641, 481), Image.NEAREST)
	poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
        print('Inference Time: ', inference_time)

        svg_canvas = svgwrite.Drawing('', size=src_size)
        for pose in poses:
            draw_pose(svg_canvas, pose, src_size, inference_box)
        svg_string = svg_canvas.tostring()
        overlay = svg2png(bytestring=svg_string)
        overlay = Image.frombytes('RGBA', (640,480), overlay, 'raw')
        # Resize both raw and overlay to 512x512 for training
        pil_image.resize((512,512), Image.NEAREST)
        overlay.resize((512,512), Image.NEAREST)
        new_im = Image.new('RGB', (total_width, 512))
        new_im.paste(pil_image, (0,0))
        new_im.paste(overlay, (512,0))
        dt_stamp = int(time.time()*10000)
        save_img_name = 'images/{}_sample.png'.format(str(dt_stamp))
        new_im.save(save_img_name)
        upload_blob(bucket_name, save_img_name, save_img_name)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main()
