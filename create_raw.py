import os
from torchvision import transforms,datasets
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt


def create_raw(image_path, raw_file):
    # image_path = "/home/yanwh/workspace/ResNet-pytorch/mammals/seal/seal-0040.jpg"
    assert os.path.exists(image_path), "image {} does not exist.".format(image_path)

    # 数据预处理操作
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 读取图片
    image = Image.open(image_path).convert("RGB")
    # plt.imshow(image)  # 展示图片

    img = data_transform(image)  # 预处理
    img = torch.unsqueeze(img, dim = 0)  # 扩展 batch 维度

    nhwc_input = np.array(img).transpose(0, 2, 3, 1)
    # nhwc_input = nhwc_input.astype(np.float16)
    nhwc_input.tofile(raw_file)
    
def prepare_img(source_dir, dest_dir):
    """
    遍历指定目录中的所有 JPG 图像文件，并将它们处理为 RAW 格式。
    
    参数:
    source_dir (str): 源图像所在的目录。
    dest_dir (str): 处理后图像的目标目录。
    """
    for root, dirs, files in os.walk(source_dir):
        for jpgs in files:
            src_image = os.path.join(root, jpgs)
            if src_image.endswith('.jpg'):
                print(src_image)
                dest_image = os.path.join(dest_dir, jpgs)
                img_filepath = os.path.abspath(src_image)
                filename, ext = os.path.splitext(dest_image)
                snpe_raw_filename = filename + '.raw'
                create_raw(src_image, snpe_raw_filename)


if __name__ == "__main__":
    input_dir = "mammals_data/calibration" 
    # class_dir = os.listdir(input_dir)
    # qnn_data_dir = "qnn_input_data_f16"

    # for class_name in class_dir:
    #     classes_dir = os.path.join(input_dir, class_name)
    #     raw_dir = os.path.join(qnn_data_dir, class_name)
    #     os.mkdir(raw_dir)
    #     prepare_img(classes_dir, raw_dir)
            
    prepare_img(input_dir, "qnn_input_data_f32")
            
