import os
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np

# 定义图片所在的文件夹路径和保存路径
input_folder = './dataset/old'  # 替换为你的图片文件夹路径
output_folder = './dataset/new_rotate2'  # 替换为你保存图片的文件夹路径
expand_width = 0

# 正方形旋转0-90度
def rotate_square_0_90():
    keyword = 'square'  # 正方形关键字
    # 创建输出文件夹（如果不存在的话）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 遍历文件夹和子文件夹
    for root, dirs, files in os.walk(input_folder):
        # 只处理包含特定关键字的子文件夹
        if keyword in os.path.basename(root):
            # 生成与当前子文件夹对应的输出文件夹路径
            relative_path = os.path.relpath(root, input_folder)
            save_folder = os.path.join(output_folder, str(relative_path+'0-90'))

            # 创建对应的输出子文件夹（如果不存在的话）
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # 处理当前子文件夹中的所有图片
            for filename in files:
                if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(root, filename)
                    # img = Image.open(img_path).convert('RGBA')
                    src = cv2.imread(img_path)
                    # 旋转0-90度并保存
                    for i in range(0, 90):
                        # 获取原图尺寸
                        (h, w) = src.shape[:2]

                        # 计算旋转后图像的尺寸
                        # 计算旋转矩阵
                        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), i, 1.0)
                        cos = np.abs(rot_mat[0, 0])
                        sin = np.abs(rot_mat[0, 1])

                        # 计算新的宽和高
                        new_w = int((h * sin) + (w * cos))
                        new_h = int((h * cos) + (w * sin))

                        # 调整旋转矩阵以考虑新的尺寸
                        rot_mat[0, 2] += new_w / 2 - w / 2
                        rot_mat[1, 2] += new_h / 2 - h / 2

                        # 使用BORDER_REFLECT将黑边填充为边缘颜色
                        dst = cv2.warpAffine(src, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                        # 保存旋转后的图片
                        cv2.imwrite(os.path.join(save_folder, f'{filename.split(".")[0]}_rotated_{i}.{filename.split(".")[-1]}'), dst)



# 三角形旋转0-120度
def rotate_triangle_0_120():
    keyword = 'triangle'  # 三角形关键字
    # 创建输出文件夹（如果不存在的话）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 遍历文件夹和子文件夹
    for root, dirs, files in os.walk(input_folder):
        # 只处理包含特定关键字的子文件夹
        if keyword in os.path.basename(root):
            # 生成与当前子文件夹对应的输出文件夹路径
            relative_path = os.path.relpath(root, input_folder)
            save_folder = os.path.join(output_folder, str(relative_path+'0-120'))

            # 创建对应的输出子文件夹（如果不存在的话）
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # 处理当前子文件夹中的所有图片
            for filename in files:
                if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(root, filename)
                    src = cv2.imread(img_path)
                    # 旋转0-90度并保存
                    for i in range(0, 90):
                        # 获取原图尺寸
                        (h, w) = src.shape[:2]

                        # 计算旋转后图像的尺寸
                        # 计算旋转矩阵
                        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), i, 1.0)
                        cos = np.abs(rot_mat[0, 0])
                        sin = np.abs(rot_mat[0, 1])

                        # 计算新的宽和高
                        new_w = int((h * sin) + (w * cos))
                        new_h = int((h * cos) + (w * sin))

                        # 调整旋转矩阵以考虑新的尺寸
                        rot_mat[0, 2] += new_w / 2 - w / 2
                        rot_mat[1, 2] += new_h / 2 - h / 2

                        # 使用BORDER_REFLECT将黑边填充为边缘颜色
                        dst = cv2.warpAffine(src, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                        # 保存旋转后的图片
                        cv2.imwrite(os.path.join(save_folder, f'{filename.split(".")[0]}_rotated_{i}.{filename.split(".")[-1]}'), dst)



rotate_square_0_90()
rotate_triangle_0_120()                        
print("图片筛选、旋转并保存完成。")
