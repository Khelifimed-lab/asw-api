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

    # 1. تحويل الصورة إلى رمادي
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. عكس الرمادي
    inv = 255 - gray

    # 3. Gaussian Blur
    blur = cv2.GaussianBlur(inv, (21, 21), 0)

    # 4. تأثير Pencil Sketch (color dodge blend)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # 5. تعزيز بسيط للتباين بدون تدمير التفاصيل
    sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)
    sketch = cv2.medianBlur(sketch, 3)  # تنعيم خفيف

    # 6. تجهيز للإرسال
    _, buf = cv2.imencode('.png', sketch)
    return send_file(io.BytesIO(buf), mimetype='image/png')
