"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""
import os
import tensorflow as tf
import xception_fused_bin
import xception_fused_bin_2
import argparse
from os.path import join
import cv2
from tensorflow.keras import backend as K
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
import numpy as np
from pathlib import Path
from dataset import image_paths, _find_images



def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def test_full_image_network(video_path, output_path,
                            start_frame=0, end_frame=None, cuda=True, video_name=None):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0] + '.avi'
    # os.makedirs(output_path, exist_ok=True)
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames", num_frames)
    if(num_frames < 100):
        return 0.0

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    #assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    #pbar = tqdm(total=end_frame - start_frame)
    avg = 0.0
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.15941712, -0.21857856, -0.14103174]),
    #                                                                     variance=np.array([2.650704 , 2.686771 , 2.5989208]))
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.0383478 , -0.08740273, -0.0162791]),
    #                                                                     variance=np.array([2.7453117, 2.7948494, 2.6936328]))
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.1031178 , -0.15620254, -0.08823884]),
    #                                                                     variance=np.array([2.643245 , 2.6748462, 2.5946023]))
    
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.14348643, -0.19796225, -0.12978919]),
    #                                                                     variance=np.array([2.6607969, 2.693048 , 2.6120377]))
    
    normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([0.03693341, -0.00503048,  0.05477703]),
                                                                         variance=np.array([2.7157836, 2.7571144, 2.668294]))

    #print("mean", normalize.mean_val)
    #print("variance", normalize.variance_val)
    model = xception_fused_bin.build_model(normalize=normalize)
    learning_rate = 2e-4
    loss = tf.keras.losses.binary_crossentropy
    metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.AUC()]

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

    load_status = model.load_weights("/workspace/data/trained_models/checkpoint/all_c23/ckpt.h5")
    total = 0
    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        if frame_num % 10 != 0:
            continue
        
        total = total + 1
        # Image size
        height, width = image.shape[:2]

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]
            cropped_face = pil_image.fromarray(cropped_face)
            cropped_face = cropped_face.resize((299, 299))
            cropped_face = np.array(cropped_face)
            cropped_face = np.expand_dims(cropped_face, axis=0)
            
            ans = model(cropped_face)
            K.clear_session()
            avg = avg + ans
        if frame_num >= end_frame:
            break
    print("total", total)
    if(total == 0):
        return 0.0
    avg = avg / total
    if(avg == 0.0):
        return 0.0
    
    print("avg act", avg.numpy()[0][0])

    return avg.numpy()[0][0]
    
    

def processFrames(args):
    print("reach")
    directory = args.video_path
    directories = sorted(map(str, filter(
        lambda x: x.is_dir(), Path(directory).iterdir())))
    for i, directory in enumerate(directories):
        images = list(sorted(image_paths(directory)))
        for image in images:
            print("Directory Name", directory)
            print("No of images", len(images))
            x = pil_image.open(image)
            x = x.resize((299, 299))
            x.save(image, 'PNG')


def countFrames(video_path):
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames", num_frames)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    # p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    #p.add_argument('--type', type=str)
    # p.add_argument('--cuda', action='store_true')
    
    args = p.parse_args()
    #if args.type == 'images':
    #    processFrames(args)
    
    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
       test_full_image_network(**vars(args))
    else:
       videos = os.listdir(video_path)
       correct_predictions = 0
       arr = []
       for video in videos:
           video_name = video.split('.mp4')
           args.video_path = join(video_path, video)
           #countFrames(args.video_path)
           score = test_full_image_network(**vars(args), video_name=video_name[0])
           if(score != 0.0):
               arr = np.insert(arr, 0, score)
           print("Predictions", arr)
       print("final arr", arr)
