#
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

import argparse
import glob
import os
import json
import numpy as np
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader
from Examples.common import image_net_config
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

def create_file_list(input_dir, output_filename, ext_pattern, print_out=False, rel_path=False):
    input_dir = os.path.abspath(input_dir)
    output_filename = os.path.abspath(output_filename)
    output_dir = os.path.dirname(output_filename)

    if not os.path.isdir(input_dir):
        raise RuntimeError('input_dir %s is not a directory' % input_dir)

    if not os.path.isdir(output_dir):
        raise RuntimeError('output_filename %s directory does not exist' % output_dir)

    glob_path = os.path.join(input_dir, ext_pattern)
    file_list = glob.glob(glob_path)

    if rel_path:
        file_list = [os.path.relpath(file_path, output_dir) for file_path in file_list]

    if len(file_list) <= 0:
        if print_out: print('No results with %s' % glob_path)
    else:
        with open(output_filename, 'w') as f:
            f.write('\n'.join(file_list))
            if print_out: print('%s created listing %d files.' % (output_filename, len(file_list)))

def save_raw_file_paths(root_dir, output_file):
    """
    从指定目录中查找所有 .raw 文件，并将它们的路径保存到一个文本文件中。

    :param root_dir: 要搜索的根目录
    :param output_file: 输出的文本文件路径
    """
    # 创建一个空列表以存储文件路径
    raw_file_paths = []

    # 遍历根目录及其所有子目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        import pdb
        pdb.set_trace()
        for filename in filenames:
            if filename.endswith('.raw'):
                # 生成文件的完整路径
                file_path = os.path.join(dirpath, filename)
                # 将路径添加到列表中
                raw_file_paths.append(file_path)

    # 将路径写入输出文件
    with open(output_file, 'w') as f:
        for path in raw_file_paths:
            f.write(path + '\n')

def main():
    parser = argparse.ArgumentParser(description="Create file listing matched extension pattern.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_dir',
                        help='Input directory to look up files.',
                        default='.')
    parser.add_argument('-o', '--output_filename',
                        help='Output filename - will overwrite existing file.',
                        default='file_list.txt')
    parser.add_argument('-e', '--ext_pattern',
                        help='Lookup extension pattern, e.g. *.jpg, *.raw',
                        default='*.raw')
    parser.add_argument('-r', '--rel_path',
                        help='Listing to use relative path',
                        action='store_true')
    args = parser.parse_args()

    # input_dir = args.input_dir
    # class_dir = os.listdir(input_dir)

    # for class_name in class_dir:
    #     classes_dir = os.path.join(input_dir, class_name)
    #     create_file_list(classes_dir, args.output_filename, args.ext_pattern, print_out=True, rel_path=args.rel_path)
    # save_raw_file_paths(args.input_dir, args.output_filename)

    data_transfrom={
    "train":transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val":transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    data_root="mammals_data" #图片根目录
    # 判断路径是否存在
    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)
    # 训练集
    val_data = datasets.ImageFolder(root=os.path.join(data_root,"val"),
                                    transform=data_transfrom["val"])
    
    class_dict = dict((v,k) for k,v in val_data.class_to_idx.items()) #{0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    # write dict into json file，将类别和索引写入josn文件
    josn_str=json.dumps(class_dict,indent=4) # indent=4控制格式缩进，一般为4或2
    with open("class_index.json","w") as josn_file:
        josn_file.write(josn_str)

    train_dataloader = DataLoader(val_data, batch_size = 10,
                                  shuffle = False, num_workers = 1)
    
    create_file_list(args.input_dir, args.output_filename, args.ext_pattern, print_out=True, rel_path=args.rel_path)


if __name__ == '__main__':
    main()
