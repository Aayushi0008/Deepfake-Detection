import os
import argparse
from os.path import join
import cv2
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


def processVideos(video_path, output_path,
                            start_frame=0, end_frame=None, cuda=True, video_name=None):
    print('Starting: {}'.format(video_path))
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0] + '.avi'
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames", num_frames)

    # face detector
    face_detector = dlib.get_frontal_face_detector()

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    frame_num = 0
    if(start_frame >= num_frames - 1):
        return
    end_frame = end_frame if end_frame else num_frames

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        
        height, width = image.shape[:2]

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # Largest face is taken
            face = faces[0]
            
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]
            cropped_face = pil_image.fromarray(cropped_face)
            cropped_face = cropped_face.resize((299, 299))
            cropped_face = np.array(cropped_face)
            cv2.imwrite(join(output_path, video_name + '_{:04d}.png'.format(frame_num)),
                        cropped_face)
        if frame_num >= end_frame:
            break

def resizeFrames(args):
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
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    #p.add_argument('--cuda', action='store_true')
    
    args = p.parse_args()
    
    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
       processVideos(**vars(args), video_name="fake")
    else:
       videos = os.listdir(video_path)
       for video in videos:
           video_name = video.split('.mp4')
           args.video_path = join(video_path, video)
           #countFrames(args.video_path)
           processVideos(**vars(args), video_name=video_name[0])
