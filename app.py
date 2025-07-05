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

    # تحليل ذكي للإضاءة والتباين
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)
    otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    # حساب عتبة البكسلات السوداء
    if contrast < 15:
        adjusted_thresh = max(0, otsu_thresh - 5)
    elif contrast > 50:
        adjusted_thresh = max(0, otsu_thresh - 20)
    else:
        adjusted_thresh = max(0, otsu_thresh - 10)

    if mean_brightness < 80:
        adjusted_thresh += 5
    elif mean_brightness > 180:
        adjusted_thresh -= 5

    black_mask = gray <= adjusted_thresh  # لحماية الأسود

    # Gaussian Blur + تأثير الرسم
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # تعزيز الأسود من الأصل
    sketch[black_mask] = gray[black_mask]

    # تحسين التباين العام (اختياري)
    sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)

    # تحويل إلى صورة قابلة للإرسال
    _, buf = cv2.imencode('.png', sketch)
    return send_file(io.BytesIO(buf), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
