import os
import struct
import numpy as np


#加载文件系统数据
def load_data(path,kind='train'):
    image_path = os.path.join(path,"%s-images.idx3-ubyte"%kind)
    label_path = os.path.join(path,"%s-labels.idx1-ubyte"%kind)
    #二进制方式读取
    with open(label_path,'rb') as lb:
        #按照官方mnist包的格式读取
        magic,n = struct.unpack('>II',lb.read(8))
        labels = np.fromfile(lb,dtype=np.uint8)
    with open(image_path,'rb') as img:
        magic, num, rows, cols = struct.unpack('>IIII',img.read(16))
        images = np.fromfile(img,dtype=np.uint8).reshape(len(labels),784)
        images[images<=127]=0
        images[images>127]=1
    return images,labels

