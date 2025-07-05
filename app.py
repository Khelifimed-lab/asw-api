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

    # تحويل إلى رمادي
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # حساب قيمة العتبة تلقائيًا
    thresh_val = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    # إنشاء قناع للبكسلات السوداء أو الداكنة حسب القيمة المحسوبة
    black_mask = gray <= thresh_val

    # تطبيق تأثير Sketch
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # استرجاع اللون الأسود الحقيقي
    sketch[black_mask] = gray[black_mask]

    _, buf = cv2.imencode('.png', sketch)
    return send_file(io.BytesIO(buf), mimetype='image/png')
