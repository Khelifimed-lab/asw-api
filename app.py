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
    mean_brightness = np.mean(gray)
    contrast = gray.std()

    # 1. Otsu threshold + تعويض ذكي
    otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    adjusted_thresh = max(0, otsu_thresh - (5 if mean_brightness < 100 else 15))
    black_mask = gray <= adjusted_thresh

    # 2. عكس الصورة
    inv = 255 - gray

    # 3. Gaussian Blur مخصص حسب الحجم
    height, width = gray.shape[:2]
    blur_strength = max(5, min(21, (width + height) // 150 | 1))
    blur = cv2.GaussianBlur(inv, (blur_strength, blur_strength), 0)

    # 4. scale حسب التباين
    if contrast < 20:
        scale = 200
    elif contrast > 50:
        scale = 300
    else:
        scale = 256

    # 5. توليد رسم سكيتش
    sketch = cv2.divide(gray, 255 - blur, scale=scale)

    # 6. إعادة البكسلات السوداء الأصلية
    sketch[black_mask] = gray[black_mask]

    _, buf = cv2.imencode('.png', sketch)
    return send_file(io.BytesIO(buf), mimetype='image/png')
