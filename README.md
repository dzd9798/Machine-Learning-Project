机器学习大作业 使用SVM以及卷积神经网络实现人脸分类

/yaleBExtData为所使用的数据集

SVM实现分类代码在/libsvm-3.24/python文件夹下 pr_svm.py为实现分类训练的主代码，使用前需要将代码中的读入路径改为数据集所在路径 visualization.py为特征提取可视化代码

神经网络实现分类代码在/CNN文件夹下 main_cnn.py为实现分类训练的主代码，运行需要tensorflow版本2.0.0 
运行main_cnn.py后训练好的模型参数在/CNN/goodmodel文件夹下，可通过CNN_visualization.py读入模型，输出每一层的结果进行可视化操作，改变代码中32行第二个参数的数值即可改变输出的层数。
