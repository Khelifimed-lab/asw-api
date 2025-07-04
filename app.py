@app.route('/sketch', methods=['POST'])
def sketch():
    img_array = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return "Invalid image", 400

    # تحويل إلى رمادي
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # تحديد البكسلات السوداء (أقل من أو يساوي 30)
    mask_black = gray <= 30

    # إنشاء صورة بيضاء بالكامل
    result = np.full_like(gray, 255)

    # إرجاع البكسلات السوداء كما هي
    result[mask_black] = gray[mask_black]

    _, buf = cv2.imencode('.png', result)
    return send_file(io.BytesIO(buf), mimetype='image/png')
