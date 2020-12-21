import argparse
import glob
import os
import sys

import cv2
import dlib
import numpy as np
import scipy.io as sio
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tqdm import tqdm

import mobilenet_v1
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
from utils.inference import crop_img, predict_dense, parse_roi_box_from_bbox
from utils.render import cget_depths_image

STD_SIZE = 120
DATA_DIR = '/mnt/79d2aab6-d2be-40fa-b606-682b8f995226/ppg/siw_data'


def resize_depth(imgdepth):
    img = np.array(imgdepth)
    res = np.zeros((32, 32, 1), dtype=float)
    for x in range(32):
        realx = 8 * x
        for y in range(32):
            realy = 8 * y
            res[x, y, 0] = np.mean(img[realx:realx + 8, realy:realy + 8])
    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    args = parser.parse_args()

    index = 0

    folder = np.load('siw_npz/folder.npz')
    label = np.load('siw_npz/label.npz')

    # Anchors = {}
    Labels_D = {}

    # 1. Enregistrement des modèles pré-entraînés
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. Function de detection des visages
    face_detector = dlib.get_frontal_face_detector()

    # 3. Detection
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    print('len(folder) :', len(folder))

    for item in tqdm(folder):
        try:
            img_ori = cv2.imread(str(folder[item]))
            rects = face_detector(img_ori, 1)

            if len(rects) != 0:
                for rect in rects:
                    bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                    roi_box = parse_roi_box_from_bbox(bbox)
                    img = crop_img(img_ori, roi_box)
                    img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                    input = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        if args.mode == 'gpu':
                            input = input.cuda()
                        param = model(input)
                        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                vertices_lst = []
                vertices = predict_dense(param, roi_box)
                vertices_lst.append(vertices)

                # Anchor = gen_anchor(param=param, kernel_size=args.paf_size)
                # Anchors[item] = Anchor

                depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1)
                if int(item) % 100 == 0:
                    print('Frame processed:', item)
                if label[item] == 0:  # real face
                    Labels_D[item] = resize_depth(depths_img)
                else:  # spoof face
                    Labels_D[item] = np.zeros((32, 32, 1), dtype=float)
            else:
                # Case our cropping didn't work
                # Anchors[item] = np.zeros((2, 4096), dtype=float)
                # print('fausse image:', item)
                Labels_D[item] = np.zeros((32, 32, 1), dtype=float)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(folder[item], ':', exc_type, fname, exc_tb.tb_lineno, str(e))

    # np.savez("anchors.npz", **Anchors)
    np.savez("siw_npz/labels_D.npz", **Labels_D)
