import os
import random
import shutil

def sample_images_from_directories(root_dir, num_samples_per_class=10, output_dir='test_images'):
    """
    从每个类别文件夹中随机抽取指定数量的图像

    :param root_dir: 包含类别文件夹的根目录
    :param num_samples_per_class: 每个类别要抽取的图像数量
    :param output_dir: 存放抽取图像的输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历根目录中的每个类别文件夹
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):  # 确保是文件夹
            # 获取该类别下的所有图片文件
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            
            # 随机抽取指定数量的图片
            sampled_images = random.sample(images, min(num_samples_per_class, len(images)))

            # 创建类别的输出文件夹
            class_output_path = os.path.join(output_dir, class_name)
            os.makedirs(class_output_path, exist_ok=True)

            # 复制抽取到的图片到输出目录
            for image in sampled_images:
                image_source_path = os.path.join(class_path, image)
                shutil.copy(image_source_path, class_output_path)

# 使用示例
root_directory = 'mammals_data/train'  # 替换为你的数据集路径
sample_images_from_directories(root_directory, num_samples_per_class=10)