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

    # تحويل الصورة إلى رمادي
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- تحليل تلقائي لخصائص الصورة ---
    mean_brightness = np.mean(gray)      # متوسط الإضاءة
    contrast = np.std(gray)              # التباين
    otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    # --- تحديد العتبة تلقائيًا حسب السطوع والتباين ---
    if contrast < 15:
        adjusted_thresh = max(0, otsu_thresh - 5)
    elif contrast > 50:
        adjusted_thresh = max(0, otsu_thresh - 20)
    else:
        adjusted_thresh = max(0, otsu_thresh - 10)

    # تعديل إضافي حسب الإضاءة
    if mean_brightness < 80:
        adjusted_thresh += 5
    elif mean_brightness > 180:
        adjusted_thresh -= 5

    # قناع البكسلات السوداء
    black_mask = gray <= adjusted_thresh

    # --- ضبط Gaussian Blur حسب حجم الصورة ---
    h, w = gray.shape
    blur_size = max(5, min(31, ((h + w) // 150) | 1))  # فردي دائمًا
    blur = cv2.GaussianBlur(255 - gray, (blur_size, blur_size), 0)

    # --- ضبط تأثير الـ sketch حسب التباين ---
    if contrast < 20:
        scale = 200
    elif contrast > 60:
        scale = 300
    else:
        scale = 256

    # تنفيذ تأثير الرسم
    sketch = cv2.divide(gray, 255 - blur, scale=scale)

    # إعادة الخطوط السوداء الأصلية كما هي
    sketch[black_mask] = gray[black_mask]

    # تجهيز الصورة للرد
    _, buf = cv2.imencode('.png', sketch)
    return send_file(io.BytesIO(buf), mimetype='image/png')
