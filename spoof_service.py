import json
import warnings
import os
import sys

import numpy as np
from flask import Flask, Response, request

from inference import infer

warnings.filterwarnings("ignore")

app = Flask(__name__)


@app.route("/spoof", methods=['POST', 'GET'])
def intro():
    return '--- Facial Spoof Detection System ---'


@app.route("/spoof/predict", methods=['POST', 'GET'])
def detect_spoof():
    data = request.json

    images = data['images']
    images = np.array(images)

    result = {}

    try:
        prediction, probability = infer(images=images)
        result['predicted_class'] = prediction
        result['probability'] = probability
        result['exception'] = None
        print(result)

        return Response(json.dumps(result, indent=4), mimetype='application/json')

    except Exception as exp:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        result = {'exception': str(exp)}
        return Response(json.dumps(result, indent=4), mimetype='application/json')


def create_service():
    return app


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
