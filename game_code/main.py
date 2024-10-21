import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import json
from tensorflow.keras.models import load_model

# 将整个文件写成一个类
class ImageRecognition:
    def __init__(self):
        self.img_size = 200
        self.num_classes = 9  # 3种颜色 × 3种形状 = 9类
        self.label_dict = {}
        self.start_fit = False # 是否开始训练模型(有模型文件就不用)
        self.image_dir = './dataset/new_rotate'  # 图片数据集路径
        self.test_status = False  # 是否测试模式（测试模式不退出，一直预测）
        self.model_name = 'shape_color_classifier.h5'    # 模型文件名
        self.take_photo_path = r'F:/Image_Recognition_Project/dataset/new_rotate/red_circle/red_circle_5.bmp' # 测试图片路径

    # 读取标签数据
    def load_label(self):
        with open('./script/chat.json', 'r') as file:
            file_content = file.read()
            self.label_dict = json.loads(file_content)

    # 加载图片数据并预处理
    def load_data(self):
        images = []
        labels = []
        for index, category in enumerate(os.listdir(self.image_dir)): # index是顺序，category是文件夹名
            category_path = os.path.join(self.image_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    images.append(img)
                    label = int(self.label_dict[category])  # 通过文件夹名获取类别
                    labels.append(label) # 注意：这里的标签是类别名，文件夹名称已经代表了类别
        images = np.array(images) / 255.0  # 归一化
        labels = np.array(labels)
        return images, labels

    # 加载模型
    def load_model(self):
        # 判断是否有模型文件
        if not os.path.exists(self.model_name):
            model = load_model('model_name')
            print('模型加载成功')
            return model
        else:
            print('模型文件不存在')
            return None

    #


# 假设你已经有图片数据集，图片大小统一为64x64
img_size = 200
num_classes = 9  # 3种颜色 × 3种形状 = 9类
label_dict = {}
start_fit = False # 是否开始训练模型(有模型文件就不用)
image_dir = './dataset/new_rotate'  # 图片数据集路径
test_status = False  # 是否测试模式（测试模式不退出，一直预测）
model_name = 'shape_color_classifier.h5'    # 模型文件名
take_photo_path = r'F:/Image_Recognition_Project/dataset/new_rotate/red_circle/red_circle_5.bmp' # 测试图片路径

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

# 加载模型
def load_model():
    # 判断是否有模型文件
    if not os.path.exists(model_name):
        model = load_model('model_name')
        print('模型加载成功')
        return model
    else:
        print('模型文件不存在')
        return None        
    

model = load_model()

# 预测函数
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = np.expand_dims(img, axis=0) / 255.0  # 归一化并添加批次维度
    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)  # 返回类别和准确性

# 示例：识别新图像
label, confidence = predict_image(take_photo_path)

