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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_mask = gray <= 40  # بدل 30 حسب تجربتك

    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # إعادة اللون الأسود النقي فقط
    sketch[black_mask] = 0

    # إزالة كل الرماديات الطفيفة: فقط أسود أو أبيض
    sketch = np.where(sketch < 30, 0, 255).astype(np.uint8)

    _, buf = cv2.imencode('.png', sketch)
    return send_file(io.BytesIO(buf), mimetype='image/png')
