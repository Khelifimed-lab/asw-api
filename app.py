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

    # نحول الصورة إلى رمادي فقط لتحليل اللون
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # نحدد كل ما هو قريب من الأسود (قيم رمادية أقل من 50)
    mask_black = gray < 50  # كل شيء غامق جدًا

    # نحول كل الصورة إلى أبيض
    result = np.full_like(gray, 255)

    # نعيد البكسلات السوداء فقط كما كانت
    result[mask_black] = gray[mask_black]

    # تحويل للتنسيق المناسب للإرجاع
    _, buf = cv2.imencode('.png', result)
    return send_file(io.BytesIO(buf), mimetype='image/png')
