from flask import Flask, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

@app.route('/sketch', methods=['POST'])
def sketch():
    img_array = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return "Invalid image", 400

    # 1. تحويل لتدرج الرمادي
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. إزالة ضوضاء بسيطة
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. تحويل إلى أبيض وأسود مع خلفية بيضاء وخطوط سوداء
    _, result = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # 4. حفظ الصورة
    _, buf = cv2.imencode('.png', result)
    return send_file(io.BytesIO(buf), mimetype='image/png')
