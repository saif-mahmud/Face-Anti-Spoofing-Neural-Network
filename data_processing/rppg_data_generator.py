import glob
import os
import sys

import cv2
import dlib
import numpy as np
import torch
from imutils import face_utils
from tqdm import tqdm

DATA_DIR = '/mnt/79d2aab6-d2be-40fa-b606-682b8f995226/ppg/siw_data'
RPPG_DIR = '/mnt/79d2aab6-d2be-40fa-b606-682b8f995226/ppg/fft_rppg'


def get_face_coordinates(face_file: str):
    with open(face_file, 'r') as f:
        face_coordinates = [[int(num) for num in line.strip().split(' ')] for line in f]

    return face_coordinates


def get_rppg_data(n_frames=5):
    FRAMES = list()
    RPPG_GT = list()

    data_dir, rppg_dir = DATA_DIR, RPPG_DIR
    face_detector = dlib.get_frontal_face_detector()

    # List all video files and corresponding face location files
    video_files = sorted(glob.glob(data_dir + '/**/*.mov', recursive=True))
    face_files = sorted(glob.glob(data_dir + '/**/*.face', recursive=True))
    rppg_files = sorted(os.listdir(rppg_dir))

    for i in tqdm(range(len(video_files))):

        video_src = video_files[i]
        face_src = face_files[i]

        vid_fname = os.path.split(video_src)[1].split('.')[0]
        rppg_fname = vid_fname + '.npy'
        if (rppg_fname not in rppg_files) and ('Live' in rppg_fname):
            continue
        else:
            if 'Live' in rppg_fname:
                rppg_gt = np.load(os.path.join(rppg_dir, rppg_fname))
            else:
                rppg_gt = np.zeros((50,))
            RPPG_GT.append(rppg_gt)

        cap = cv2.VideoCapture(video_src)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        face_coordinates = get_face_coordinates(face_src)
        currentFrame, frame_idx = 0, 0

        while (currentFrame < n_frames) and (frame_idx < length):
            try:
                ret, frame = cap.read()

                left, top, right, bottom = face_coordinates[frame_idx]
                face_frame = frame[top:bottom, left:right]

                frame_idx += 1

                if (face_frame.shape[0] == 0) or (face_frame.shape[1] == 0):
                    rects = face_detector(frame, 1)
                    if len(rects) > 0:
                        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
                        face_frame = frame[y:y + h, x:x + w]
                    else:
                        continue

                face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                face_frame = cv2.resize(face_frame, (256, 256), interpolation=cv2.INTER_NEAREST)
                face_frame = np.divide(face_frame, 255)
                FRAMES.append(face_frame)

                currentFrame += 1

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(video_src, '-', currentFrame, ':', exc_type, fname, exc_tb.tb_lineno)
                continue

        cap.release()
        cv2.destroyAllWindows()

    return np.array(FRAMES), np.array(RPPG_GT)


def get_rppg_dataloader():
    img_file = 'data_processing/img_data.npy'
    rppg_file = 'data_processing/rppg.npy'

    print(img_file, os.path.exists(img_file))
    print(rppg_file, os.path.exists(rppg_file))

    if os.path.exists(img_file) and os.path.exists(rppg_file):
        img_data = np.load(img_file)
        rppg_data = np.load(rppg_file)
    else:
        img_data, rppg_data = get_rppg_data()
        np.save(img_file, img_data)
        np.save(rppg_file, rppg_data)

    print(img_data.shape)
    print(rppg_data.shape)

    img_dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.transpose(img_data, (0, 3, 1, 2)), dtype=torch.float32))
    img_dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=5, shuffle=False, drop_last=True)

    rppg_data = torch.tensor(rppg_data, dtype=torch.float32)

    return img_dataloader, rppg_data
