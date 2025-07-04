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

    # 2. حفظ البكسلات الداكنة جدًا (سوداء تقريبًا)
    black_mask = gray <= 50   # يمكنك تعديل القيمة حسب شدة اللون الأسود الذي تريد الحفاظ عليه

    # 3. تنفيذ تأثير الـ sketch (dodge blend)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # 4. إعادة إدراج البكسلات السوداء الأصلية كما هي
    sketch[black_mask] = gray[black_mask]

    # 5. حفظ الصورة وإرسالها
    _, buf = cv2.imencode('.png', sketch)
    return send_file(io.BytesIO(buf), mimetype='image/png')
