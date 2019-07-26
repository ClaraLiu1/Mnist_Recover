#分解weight  然后变成矩阵相乘形式 再合并这些矩阵
#得到output 进行分解 合并矩阵

import numpy as np
import tensorflow as tf


#单个weight添零变成一个矩阵，可与输入进行矩阵乘法得到输出（输入与输出通通转换为列vector的方法）


#weight1 也就是对于第一个卷积层结果与卷积核的恢复
#首先得到weight1的信息
#分解成一个个weight
#添零变成一个矩阵
#纵向合并成一个矩阵
#求伪逆Ct



#输出内容的unpickle
#分解为单个输出
#合并输出，并转为列向量


#Ct与输出的列向量矩阵相乘得到结果
#结果存为图片并显示