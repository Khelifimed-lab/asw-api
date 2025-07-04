@app.route('/sketch', methods=['POST'])
def sketch():
    img_array = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return "Invalid image", 400

    # تحويل لصورة رمادية
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)

    # توليد الرسم بخط ناعم
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # تعزيز طفيف للتباين إذا رغبت
    sketch = np.clip(sketch * 1.2, 0, 255).astype(np.uint8)

    _, buf = cv2.imencode('.png', sketch)
    return send_file(io.BytesIO(buf), mimetype='image/png')
