import glob
import os
import sys

import cv2
import dlib
import numpy as np
from PIL import Image
from imutils import face_utils
from tqdm import tqdm

DATA_DIR = '/mnt/79d2aab6-d2be-40fa-b606-682b8f995226/ppg/siw_data'


# DATA_DIR = '/home/tigerit/saif/siw_data'
# VAL_DIR = '/mnt/79d2aab6-d2be-40fa-b606-682b8f995226/ppg/siw_val'
# DEPTH_DIR = '/mnt/79d2aab6-d2be-40fa-b606-682b8f995226/ppg/kaggle/siw_depth_maps'


def get_face_coordinates(face_file: str):
    with open(face_file, 'r') as f:
        face_coordinates = [[int(num) for num in line.strip().split(' ')] for line in f]

    return face_coordinates


def get_label(video_src: str):
    fname = os.path.split(video_src)[1]
    label = fname.split('_')[0]

    return label


def label_encoder(label: str, binary=True):
    if binary:
        if label == 'Live':
            return 0
        else:
            return 1
    else:
        label_map = {'Live': 0, 'Makeup': 1, 'Mask': 2, 'Paper': 3, 'Partial': 4, 'Replay': 5}
        return label_map[label]


def resize_center_eyes(frame, lx, ly, rx, ry):
    img = np.array(frame).transpose(1, 0, 2)
    res = np.zeros((256, 256, 3), dtype=float)
    center = (0.5 * (lx + rx), 0.5 * (ly + ry))

    for x in range(256):

        realx = int((x - 128) * 3 + center[0])

        for y in range(256):
            realy = int((y - 100) * 3 + center[1])
            res[x, y, :] = np.mean(np.mean(img[realx - 1:realx + 2, realy - 1:realy + 2, ::-1], axis=0), axis=0)

    res = res.transpose(1, 0, 2)
    img_png = res

    return res / 255, Image.fromarray(img_png.astype('uint8'))


def siw_data_process_1(n_frames=5):
    folder = {}
    images = {}
    label = {}
    index = 0

    data_dir = DATA_DIR
    face_detector = dlib.get_frontal_face_detector()

    # List all video files and corresponding face location files
    video_files = sorted(glob.glob(data_dir + '/**/*.mov', recursive=True))
    face_files = sorted(glob.glob(data_dir + '/**/*.face', recursive=True))

    for i in tqdm(range(len(video_files))):
        try:
            video_src = video_files[i]
            face_src = face_files[i]

            cap = cv2.VideoCapture(video_src)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            face_coordinates = get_face_coordinates(face_src)

            currentFrame, frame_idx = 0, 0

            while (currentFrame < n_frames) and (frame_idx < length):
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

                try:
                    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                    face_frame = cv2.resize(face_frame, (256, 256), interpolation=cv2.INTER_NEAREST)
                except Exception as exp:
                    print('[Exception : face frame] :', frame_idx, ':', str(exp))
                    continue

                pil_face_frame = Image.fromarray(face_frame.astype('uint8'))
                face_frame = np.divide(face_frame, 255)

                images[str(index)] = face_frame

                video_fname, _ = os.path.splitext(video_src)
                name = video_fname + '_' + str(currentFrame) + '.png'
                folder[str(index)] = name

                pil_face_frame.save(name)
                label[str(index)] = label_encoder(get_label(video_src))

                index += 1
                currentFrame += 1

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(video_src, ':', exc_type, fname, exc_tb.tb_lineno)

        cap.release()
        cv2.destroyAllWindows()

    np.savez("siw_npz/images.npz", **images)
    np.savez("siw_npz/label.npz", **label)
    np.savez("siw_npz/folder.npz", **folder)


if __name__ == '__main__':
    siw_data_process_1()
