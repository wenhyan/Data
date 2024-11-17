import torch
import onnx
import numpy as np
import onnxruntime as ort
from torchvision import transforms,models
import os
from PIL import Image
import matplotlib.pyplot as plt
# from models import resnet18
import json
    

def main():
    
    model_path = "resnet18.onnx"
    ort_session = ort.InferenceSession(model_path)
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_path = "mammals_data/train/vampire_bat/vampire_bat-0005.jpg"
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
    plt.imshow(image)  # 展示图片

    img = data_transform(image)  # 预处理

    img = torch.unsqueeze(img, dim = 0)  # 扩展 batch 维度
    
    onnx_input = img.numpy()
    
    onnx_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: onnx_input})

    # 读取类别文件 json
    json_path = "class_index.json"  # 确保正确拼写
    assert os.path.exists(json_path), "json {} does not exist.".format(json_path)
    with open(json_path) as json_file:
        cls_index = json.load(json_file)

    # 加载模型
    net = models.resnet18(num_classes = 45)
    weight_path = "resnet18.pth"
    assert os.path.exists(weight_path), "weight {} does not exist.".format(weight_path)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.to(device)
    net.eval()

    qnn_f32_output = np.fromfile("f32_output/Result_0/output.raw", np.float32)
    qnn_8bit_output = np.fromfile("8bit_output/Result_3/output.raw", np.float32)
    qnn_f16_output = np.fromfile("f16_output/Result_0/output.raw", np.float16)

    with torch.no_grad():
        output = torch.squeeze(net(img.to(device))).cpu()
        onnx_output = onnx_outputs[0]
        onnx_output = onnx_output.reshape(45)
        onnx_output = torch.tensor(onnx_output)
        
        qnn_f32_output = qnn_f32_output.reshape(45)
        qnn_f32_output = torch.tensor(qnn_f32_output)

        # qnn_f16_output = qnn_f16_output.astype(np.float32)
        qnn_f16_output = qnn_f16_output.reshape(45)
        qnn_f16_output = torch.tensor(qnn_f16_output)
        
        qnn_8bit_output = qnn_8bit_output.reshape(45)
        qnn_8bit_output = torch.tensor(qnn_8bit_output)
        
        pre = torch.softmax(qnn_f16_output, dim = 0)  # 计算概率
        cls = torch.argmax(pre).numpy()  # 预测的类别的索引

    print_res = "class: {}   prob: {:.3}".format(cls_index[str(cls)], pre[cls].numpy())
    plt.title(print_res)  # 将结果写在图片的标题上

    for i in range(len(pre)):
        print("class: {:10} prob: {:.3}".format(cls_index[str(i)], pre[i].numpy()))

    plt.show()

if __name__ == "__main__":
    main()