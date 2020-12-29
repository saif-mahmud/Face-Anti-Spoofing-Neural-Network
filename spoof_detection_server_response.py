import json
import os
from datetime import datetime
from multiprocessing import Process, Queue

import cv2
import face_recognition
import numpy as np
import requests

BATCH_SIZE = 5

save_dir = 'results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def producer(q):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="hog")
        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]

            image = frame[top:bottom, left:right]
            image = cv2.resize(image, (256, 256))

            q.put(image)

            frame = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
            frame = cv2.putText(frame, 'No Face Detected', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def consumer(q):
    # url = 'https://tracker.onudaan.com:5000/common/spoof/predict' # public [doesn't work]
    url = 'http://192.168.1.199:5000/common/spoof/predict' # private
    images = list()

    while True:

        if not q.empty():
            image = q.get()
            images.append(image)

            if len(images) == BATCH_SIZE:
                images = np.array(images)
                print(images.shape)

                payload = {'images': images.tolist()}
                r = requests.post(url, json=payload)
                print(r.status_code)

                parsed = json.loads(r.content)
                print(json.dumps(parsed, indent=4, sort_keys=True))

                images = list()

                timestamp = datetime.now()

                if r.json()['exception'] is None:
                    labels = r.json()['predicted_class']
                    probs = r.json()['probability']

                    for i in range(len(labels)):
                        probability = str(round(float(probs[i]), 5))
                        fname = labels[i] + '_' + timestamp.strftime("%d%m%Y_%H%M%S_%f") + '_' + probability + '.png'
                        cv2.imwrite(os.path.join(save_dir, fname), image)

                else:
                    fname = 'exception_' + timestamp.strftime("%d%m%Y_%H%M%S_%f") + '.png'
                    cv2.imwrite(os.path.join(save_dir, fname), image)


if __name__ == '__main__':
    q = Queue()
    p = Process(target=producer, args=(q,))
    c = Process(target=consumer, args=(q,))
    p.start()
    c.start()
    p.join()
    c.join()