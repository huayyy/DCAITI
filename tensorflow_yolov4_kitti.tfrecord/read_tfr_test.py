# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:44:47 2021

@author: zicec
"""
import tensorflow.compat.v1 as tf
import io
import numpy as np
import PIL.Image as pil
import readtfr as rtf
import cv2

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

classes = read_class_names("C:/Users/zicec/Documents/SS2021/DCAITI PJ/codes/data/kitti.names")
new_dict = {v : k for k, v in classes.items()}
#反转字典，输出class number

root_path = 'C:/Users/zicec/Documents/SS2021/DCAITI PJ/codes/data/kitti_tfrecords/'
tfrecord_filename = root_path + 'train.tfrecord'
dataset = tf.data.TFRecordDataset(tfrecord_filename)
dataset = dataset.map(rtf.pares_tf)
iterator = dataset.make_one_shot_iterator()

"""

写一个txt文件的annotations

"""

m=0
try:
    while True:
        image, anno = rtf.parse_record(iterator.get_next(), new_dict)
        # print(image)
        print("第",m,"组数据")
        print(anno)  
        with open('kitti_train_set.txt','a') as f:    #设置文件对象
            f.write(anno)                             #将字符串写入文件中
            f.write('\n')
        m+=1
except tf.errors.OutOfRangeError:
    print("OutOfRangeError")



"""

从tfrecord里解析每条record

"""
# m=0
# while True:
#     if m<2:
#         image, anno = rtf.parse_record(iterator.get_next(), new_dict)
#         # print(image)
#         print("第",m,"组数据")
#         print(anno)  
#         with open('data.txt','a') as f:    #设置文件对象
#             f.write(anno)                 #将字符串写入文件中
#             f.write('\n')
#         m+=1
#     else:
#         break

# anno = []

# string = ""
# sumstring = ""
# c = 0

# for parsed_record in dataset.take(2):
#     c+=1
#     array_pr = rtf.tensor_to_array(parsed_record)
#     num = len(array_pr[1])
    
#     for ii in range(num):
        
#         string += " {},{},{},{},{}".format(
#                                 array_pr[4][ii],
#                                 array_pr[6][ii],
#                                 array_pr[5][ii],
#                                 array_pr[7][ii],
#                                 new_dict[str(array_pr[1][ii], encoding = "utf-8")],
#                             )
#     sumstring = string
# print("前",c,"组数据")
# anno.append(sumstring)
# print(anno)

    

