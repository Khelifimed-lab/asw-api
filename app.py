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

    # 1. تحويل إلى رمادي
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. إنشاء نسخة نظيفة: نحافظ على الأسود والرمادي الداكن فقط
    # أي بكسل لونه أفتح من 200 → نخليه أبيض
    cleaned = np.where(gray > 200, 255, gray).astype(np.uint8)

    # 3. إزالة النقاط الصغيرة (ضوضاء) عبر فلتر خفيف
    cleaned = cv2.medianBlur(cleaned, 3)

    _, buf = cv2.imencode('.png', cleaned)
    return send_file(io.BytesIO(buf), mimetype='image/png')
