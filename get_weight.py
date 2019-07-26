from tensorflow.python import pywrap_tensorflow
import os
import pickle as pk
import tensorflow as tf
'''
用于生成weight的pk文件  
用于在后面生成图像
'''
path='/home/LeNet5_MNIST_2/weights/'

model_path='checkpoint/variable.ckpt'
reader=pywrap_tensorflow.NewCheckpointReader(model_path)
var_to_shape_map=reader.get_variable_to_shape_map()
# print(reader.get_tensor(key))
x=0
if not os.path.isdir(path):
    os.makedirs(path)
for key in var_to_shape_map:
    print('tensor_name:',key," ",reader.get_tensor(key).shape)
    if key=='Lenet/conv1/weights':
        weight=reader.get_tensor(key)

        #print('weight',weight)   #显示weight信息

        f = open(path + 'weight1', 'wb')
        pk.dump(weight, f)
        f.close()
        print("1")
        x=x+1
    elif key=='Lenet/conv3/weights' :
        weight = reader.get_tensor(key)

        #print('weight', weight)  # 显示weight信息

        f = open(path + 'weight3', 'wb')
        pk.dump(weight, f)
        f.close()
        print("2")
        x=x+1
    elif key == 'Lenet/conv5/weights':
        weight = reader.get_tensor(key)
        f = open(path + 'weight5', 'wb')
        pk.dump(weight, f)
        f.close()
        print("3")
        x=x+1
    elif x==3:
        print(x)
        break
    else:
        continue