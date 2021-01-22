import csv
import os
import warnings

import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

RPPG_DIR = '/mnt/79d2aab6-d2be-40fa-b606-682b8f995226/ppg/kaggle/SSA_RPPG_GT/ssa_rppg_siw'
SAVE_DIR = 'ssa_gt_norm'


def get_heartrate(ppg_signal, sample_rate):
    try:
        _, measures = hp.process(ppg_signal, sample_rate)
        hr = measures['bpm']
        return hr

    except Exception as exc:
        return 'HeartPy Exception'


def ssa_norm():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    rppg_files = sorted(os.listdir(RPPG_DIR))

    for i in tqdm(range(len(rppg_files))):
        rppg_fname = rppg_files[i]
        ppg_ssa = np.load(os.path.join(RPPG_DIR, rppg_fname))

        norm_ssa = np.linalg.norm(ppg_ssa)
        hr_ssa = get_heartrate(ppg_ssa, sample_rate=30.0)

        with open('ssa_norm_hr_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([rppg_fname, str(norm_ssa), str(hr_ssa)])

        fig, ax = plt.subplots()
        ax.plot(ppg_ssa)

        plt.grid(True)
        plt.ylabel('SSA rPPG Signal')
        plt.xlabel('BPM : ' + str(hr_ssa))
        _title = '[SSA-rPPG] L2 Norm : ' + str(round(norm_ssa, 5))
        ax.set_title(_title)

        out_fname = os.path.splitext(rppg_fname)[0]
        plt.savefig(os.path.join(SAVE_DIR, str(out_fname + '.png')))


if __name__ == '__main__':
    ssa_norm()
