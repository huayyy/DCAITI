import io
import os

import numpy as np
import PIL.Image as pil
from PIL import Image
import tensorflow as tf

import feature_parser
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='C:/Users/zicec/Documents/SS2021/DCAITI PJ/dataset/',
                    help='kitti数据集的位置')
parser.add_argument('--output_path', type=str, default='C:/Users/zicec/Documents/SS2021/DCAITI PJ/codes/data/kitti_tfrecords/',
                    help='TFRecord文件的输出位置')
parser.add_argument('--validation_set_size', type=int, default=2000,
                    help='验证集数据集使用大小')

def convert_kitti_to_tfrecords(data_dir, output_path, validation_set_size):
    """
    将KITTI detection 转换成TFRecords.
    :param data_dir: 源数据目录
    :param output_path: 输出文件目录
    :param validation_set_size: 验证集大小
    :return:
    """
    train_count = 0
    val_count = 0

    # 1、创建KITTI训练和验证集的tfrecord位置
    # 标注信息位置
    annotation_dir = os.path.join(data_dir,
                                  'training',
                                  'label_2')

    # 图片位置
    image_dir = os.path.join(data_dir,
                             'data_object_image_2',
                             'training',
                             'image_2')
    
    it = open(data_dir+'train.txt')
    train_text = it.read().splitlines()
    train_writer = tf.io.TFRecordWriter(output_path + 'train.tfrecord')
    val_writer = tf.io.TFRecordWriter(output_path + 'val.tfrecord')

    # 2、列出所有的图片，进行每张图片的内容和标注信息的获取，写入到tfrecords文件
    images = sorted(os.listdir(image_dir))
    for img in images:

        # （1）获取当前图片的编号数据，并拼接读取相应标注文件
        img_num = str(int(img.split('.')[0])).zfill(6)

        # （2）读取标签文件函数
        # 整数需要进行填充成与标签文件相同的6位字符串
        
        img_anno = read_annotation_file(os.path.join(annotation_dir,
                                                     img_num + '.txt'))
      
        # （3）写入训练和验证集合TFRecord文件
        # 读取拼接的图片路径，然后与过滤之后的标注结果进行合并到一个example中
        image_path = os.path.join(image_dir, img)
        example = prepare_example(image_path, img_anno)
        # 判断写入训练集还是验证集
        if img_num in train_text:
            train_writer.write(example.SerializeToString())
            train_count += 1
        else:
            val_writer.write(example.SerializeToString())
            val_count += 1

    train_writer.close()
    val_writer.close()


def read_annotation_file(filename):
    """
    读取标签文件函数
    """
    with open(filename) as f:
        content = f.readlines()
    # 分割解析内容
    content = [x.strip().split(' ') for x in content]
    # 保存内容到字典结构
    anno = dict()
    anno['type'] = np.array([x[0].lower() for x in content])
    anno['truncated'] = np.array([float(x[1]) for x in content])
    anno['occluded'] = np.array([int(x[2]) for x in content])
    anno['alpha'] = np.array([float(x[3]) for x in content])

    anno['2d_bbox_left'] = np.array([float(x[4]) for x in content])
    anno['2d_bbox_top'] = np.array([float(x[5]) for x in content])
    anno['2d_bbox_right'] = np.array([float(x[6]) for x in content])
    anno['2d_bbox_bottom'] = np.array([float(x[7]) for x in content])
    return anno


def prepare_example(image_path, annotations):
    """
    对一个图片的Annotations转换成tf.Example proto.
    :param image_path:
    :param annotations:
    :return:
    """
    # 1、读取图片内容，转换成数组格式
    with open(image_path, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = pil.open(encoded_png_io)
    image = np.asarray(image)
    
    # 进行坐标处理
    width = int(image.shape[1])
    height = int(image.shape[0])
    # 存储极坐标归一化格式
    # xmin_norm = annotations['2d_bbox_left'] / float(width)
    # ymin_norm = annotations['2d_bbox_top'] / float(height)
    # xmax_norm = annotations['2d_bbox_right'] / float(width)
    # ymax_norm = annotations['2d_bbox_bottom'] / float(height)
    xmin_norm = annotations['2d_bbox_left']
    ymin_norm = annotations['2d_bbox_top']
    xmax_norm = annotations['2d_bbox_right']
    ymax_norm = annotations['2d_bbox_bottom']
    
    
    classes_text = [x.encode('utf8') for x in annotations['type']]

    # 3、构造协议example
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': feature_parser.int64_feature(height),
        'width': feature_parser.int64_feature(width),
        'filename': feature_parser.bytes_feature(image_path.encode('utf8')),
        'encoded': feature_parser.bytes_feature(encoded_png),
        'format': feature_parser.bytes_feature('png'.encode('utf8')),
        'class': feature_parser.bytes_list_feature(classes_text),
        'truncated': feature_parser.float_list_feature(annotations['truncated']),
        'occluded': feature_parser.int64_list_feature(annotations['occluded']),
        'alpha': feature_parser.float_list_feature(annotations['alpha']),
        'xmin': feature_parser.float_list_feature(xmin_norm),
        'xmax': feature_parser.float_list_feature(xmax_norm),
        'ymin': feature_parser.float_list_feature(ymin_norm),
        'ymax': feature_parser.float_list_feature(ymax_norm)
    }))

    return example

def main(args):

    convert_kitti_to_tfrecords(
        data_dir=args.data_dir,
        output_path=args.output_path,
        validation_set_size=args.validation_set_size)


if __name__ == '__main__':

    args = parser.parse_args(sys.argv[1:])
       
    main(args)
