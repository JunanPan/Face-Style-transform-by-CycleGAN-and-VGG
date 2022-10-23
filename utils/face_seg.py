import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile#tf读写操作文件的工具接口


curPath = os.path.abspath(os.path.dirname(__file__))


class FaceSeg:
    def __init__(self, model_path=os.path.join(curPath, 'seg_model_384.pb')):#seg_model_384.pb头像分割模型
        config = tf.ConfigProto()#tf.ConfigProto()主要的作用是配置tf.Session的运算方式,比如gpu运算或者cpu运算
        config.gpu_options.allow_growth = True
        self._graph = tf.Graph()#tf.Graph()表示实例化一个用于tensorflow计算和表示用的数据流图,不负责运行计算
        self._sess = tf.Session(config=config, graph=self._graph)

        self.pb_file_path = model_path#得到模型目录
        self._restore_from_pb()#导入模型
        self.input_op = self._sess.graph.get_tensor_by_name('input_1:0')
        self.output_op = self._sess.graph.get_tensor_by_name('sigmoid/Sigmoid:0')

    def _restore_from_pb(self):#读取参数和模型的一套工具函数写法
        with self._sess.as_default():
            with self._graph.as_default():
                with gfile.FastGFile(self.pb_file_path, 'rb') as f:
                    graph_def = tf.GraphDef()#用于模型保存、恢复、读取
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')

    #处理输入图像
    def input_transform(self, image):
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)#对输入图像进行resize
        image_input = (image / 255.)[np.newaxis, :, :, :]#再标准化并添加一个维度
        return image_input
    #处理输出图像
    def output_transform(self, output, shape):#进行resize和标准化操作
        output = cv2.resize(output, (shape[1], shape[0]))
        image_output = (output * 255).astype(np.uint8)
        return image_output
    #得到截取分割后的头像区域
    def get_mask(self, image):
        image_input = self.input_transform(image)#处理输入图像
        output = self._sess.run(self.output_op, feed_dict={self.input_op: image_input})[0]#处理后送入送入tf模型
        return self.output_transform(output, shape=image.shape[:2])#返回处理后的输出图像
