from flask import Flask, render_template, request
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# 加载模型
model = load_model('shape_color_classifier.h5')

# 读取标签数据
with open('./script/chat.json', 'r') as file:
    label_dict = json.load(file)

img_size = 200  # 根据需要调整图像大小

# 预测函数
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = np.expand_dims(img, axis=0) / 255.0  # 归一化并添加批次维度
    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)  # 返回类别和准确性

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # 保存上传的图片到 static/uploads
        img_path = f"./static/uploads/{file.filename}"
        file.save(img_path)

        label, confidence = predict_image(img_path)
        result = label_dict[str(label)]
        image_path = img_path  # 保存上传的图片路径

    return render_template('index.html', label=result, confidence=confidence, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
