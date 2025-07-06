# app.py
from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return "API is running!"

@app.route('/sketch', methods=['POST'])
def sketch():
    data = request.json
    image_b64 = data.get('image_base64')

    if not image_b64:
        return jsonify({"error": "No image_base64 provided"}), 400

    # Decode base64 to image
    try:
        image_data = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": "Image decode failed", "details": str(e)}), 500

    # Convert to sketch
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)

        _, buffer = cv2.imencode('.jpg', sketch)
        output_b64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        return jsonify({"error": "Sketch processing failed", "details": str(e)}), 500

    return jsonify({"sketch_base64": output_b64})

if __name__ == '__main__':
    app.run(debug=True)
