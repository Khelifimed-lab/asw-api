from flask import Flask, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

@app.route('/')
def index():
    return "Coloring Page Creator API is running!"

@app.route('/sketch', methods=['POST'])
def sketch():
    # ✅ قراءة الصورة (من form-data أو raw body)
    if 'file' in request.files:
        file = request.files['file']
        img_array = np.frombuffer(file.read(), np.uint8)
    else:
        img_array = np.frombuffer(request.data, np.uint8)

    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "❌ لم يتم استلام صورة صالحة"}, 400

    # 🔹 تحويل إلى Sketch
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # إرجاع النتيجة كـ PNG
    ok, buf = cv2.imencode('.png', sketch)
    if not ok:
        return {"error": "❌ فشل في معالجة الصورة"}, 500

    return send_file(io.BytesIO(buf), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
