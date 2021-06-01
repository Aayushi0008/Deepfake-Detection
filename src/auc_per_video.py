import os
import tensorflow as tf
import xception_fused_bin
import argparse
from os.path import join
import cv2
from tensorflow.keras import backend as K
import dlib
from PIL import Image as pil_image
from tqdm import tqdm
import numpy as np
from pathlib import Path
from dataset import image_paths, _find_images



def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def compute_auc_per_video(video_path, output_path,
                            start_frame=0, end_frame=None, cuda=True, video_name=None):
    print('Starting: {}'.format(video_path))

    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0] + '.avi'
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames", num_frames)
    
    if(num_frames < 100):
        return 0.0

    # face detector
    face_detector = dlib.get_frontal_face_detector()

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    frame_num = 0
    #assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    avg = 0.0
    
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([0.03693341, -0.00503048,  0.05477703]),
    #                                                                     variance=np.array([2.7157836, 2.7571144, 2.668294]))

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
    model_path = args.model_path
    load_status = model.load_weights(model_path)
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
        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        
        if len(faces):
            # largest face taken
            face = faces[0]
            
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


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    #p.add_argument('--cuda', action='store_true')
    
    args = p.parse_args()
    
    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
       compute_auc_per_video(**vars(args))
    else:
       videos = os.listdir(video_path)
       correct_predictions = 0
       arr = []
       for video in videos:
           video_name = video.split('.mp4')
           args.video_path = join(video_path, video)
           score = compute_auc_per_video(**vars(args), video_name=video_name[0])
           if(score != 0.0):
               arr = np.insert(arr, 0, score)
           print("Predictions", arr)
       print("final arr", arr)
