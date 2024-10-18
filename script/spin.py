import os
from PIL import Image, ImageOps, ImageFilter

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
                    img = Image.open(img_path).convert('RGBA')

                    # 旋转0-90度并保存
                    for i in range(0, 90):
                        rotated_img = img.rotate(i, expand=True, fillcolor=None)
                        # 提取边缘颜色用于填充
                        width, height = rotated_img.size
                        left_edge = rotated_img.crop((0, 0, 1, height)).resize((50, height))  # 左边扩展
                        right_edge = rotated_img.crop((width - 1, 0, width, height)).resize((50, height))  # 右边扩展
                        top_edge = rotated_img.crop((0, 0, width, 1)).resize((width, 50))  # 上边扩展
                        bottom_edge = rotated_img.crop((0, height - 1, width, height)).resize((width, 50))  # 下边扩展

                        # 创建新图像用于放置扩展部分
                        expanded_width = width + 100
                        expanded_height = height + 100
                        expanded_img = Image.new('RGB', (expanded_width, expanded_height))

                        # 填充扩展部分
                        expanded_img.paste(top_edge, (50, 0))  # 上部扩展
                        expanded_img.paste(bottom_edge, (50, height + 50))  # 下部扩展
                        expanded_img.paste(left_edge, (0, 50))  # 左部扩展
                        expanded_img.paste(right_edge, (width + 50, 50))  # 右部扩展
                        
                        # 将旋转后的图像放置在中心
                        expanded_img.paste(rotated_img, (50, 50), rotated_img)

                        expanded_img.save(os.path.join(save_folder, f'{filename.split(".")[0]}_rotated_{i}.{filename.split(".")[-1]}'))

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
                    img = Image.open(img_path)

                    # 旋转0-119度并保存
                    for i in range(0, 120):
                        img_i = img.rotate(i, expand=True)
                        # 提取图像的边缘像素，用于边界扩展的填充
                        expanded_img = ImageOps.expand(img_i, border=expand_width, fill=0)
                        # 将扩展部分应用模糊效果，创造自然的扩展边缘
                        blurred_img = expanded_img.filter(ImageFilter.GaussianBlur(radius=expand_width // 2))
                        blurred_img.save(os.path.join(save_folder, f'{filename.split(".")[0]}_rotated_{i}.{filename.split(".")[-1]}'))

rotate_square_0_90()
rotate_triangle_0_120()                        
print("图片筛选、旋转并保存完成。")
