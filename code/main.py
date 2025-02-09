import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import json
from tensorflow.keras.models import load_model

# 假设你已经有图片数据集，图片大小统一为64x64
img_size = 200
num_classes = 9  # 3种颜色 × 3种形状 = 9类
label_dict = {}
start_fit = False # 是否开始训练模型(有模型文件就不用)
image_dir = './dataset/new_rotate'  # 图片数据集路径
test_status = False  # 是否测试模式（测试模式不退出，一直预测）

# 读取标签数据
with open('./script/chat.json', 'r') as file:
    file_content = file.read()
    label_dict = json.loads(file_content)

# 加载图片数据并预处理
def load_data(img_dir):
    images = []
    labels = []
    for index, category in enumerate(os.listdir(img_dir)): # index是顺序，category是文件夹名
        category_path = os.path.join(img_dir, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                label = int(label_dict[category])  # 通过文件夹名获取类别
                labels.append(label) # 注意：这里的标签是类别名，文件夹名称已经代表了类别
    images = np.array(images) / 255.0  # 归一化
    labels = np.array(labels)
    return images, labels

# 假设图片保存在 'data/' 文件夹下，每个子文件夹代表一种类别
images, labels = load_data(image_dir)

# 训练模型
def train_model():
    # 不用重新训练模型
    if not start_fit:
        model = load_model('shape_color_classifier.h5')
        return model
    else:   # 重新训练模型
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        # 将标签转换为one-hot编码
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        # 构建卷积神经网络模型
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')  # 输出9类
        ])

        # 编译模型
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # 训练模型
        model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test)) # validation_data用于验证集

        # 保存模型
        model.save('shape_color_classifier.h5')
        return model

model = train_model()

# 预测函数
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = np.expand_dims(img, axis=0) / 255.0  # 归一化并添加批次维度
    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)  # 返回类别和准确性

# 示例：识别新图像
label, confidence = predict_image(r'F:\Image_Recognition_Project\dataset\new_rotate\red_circle\red_circle_5.bmp')
print(f'识别结果: {label_dict[str(label)]}, 准确率: {confidence}')
