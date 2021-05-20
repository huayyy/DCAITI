import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
import cv2


root_path = 'C:/Users/zicec/Documents/SS2021/DCAITI PJ/codes/data/kitti_tfrecords/'
tfrecord_filename = root_path + 'train.tfrecord'

def pares_tf(example_proto):
    #定义解析的字典
    dics = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'filename': tf.FixedLenFeature([], tf.string),                           
        'encoded': tf.FixedLenFeature([], tf.string),
        'format': tf.FixedLenFeature([], tf.string),                               
        'class': tf.FixedLenSequenceFeature([], tf.string,allow_missing=True),
        'truncated': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'occluded': tf.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
        'alpha': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'xmin': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'xmax': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'ymin': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'ymax': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True)
    }
    #调用接口解析一行样本
    parsed_example = tf.parse_single_example(example_proto,dics)
    image = tf.image.decode_png(parsed_example['encoded'], channels=3)
    #image = tf.image.resize(image, (416, 416))
    #image = (tf.cast(image,tf.float32)/255.0)
    filename = parsed_example['filename']
    label = parsed_example['class']
    width = parsed_example['width']
    height = parsed_example['height']
    xmin = parsed_example['xmin']
    xmax = parsed_example['xmax']
    ymin = parsed_example['ymin']
    ymax = parsed_example['ymax']
    image = tf.image.resize(image, (height,width))
    image = (tf.cast(image,tf.float32)/255.0)
    return image,label,width,height,xmin,xmax,ymin,ymax,filename


def tensor_to_array(parsed_record):
    array = []
    for it in range(len(parsed_record)):
        array.append(parsed_record[it].numpy())
    return array


def parse_record(parsed_record,dict):    
    
    array_pr = tensor_to_array(parsed_record)
    num = len(array_pr[1])  
    image = cv2.cvtColor(array_pr[0], cv2.COLOR_BGR2RGB)
    string = ""
    for ii in range(num):            
        string += " {},{},{},{},{}".format(
                                array_pr[4][ii],
                                array_pr[6][ii],
                                array_pr[5][ii],
                                array_pr[7][ii],
                                dict[str(array_pr[1][ii], encoding = "utf-8")],
                            )
    string = str(array_pr[8], encoding = "utf-8")+string
    return image, string

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
