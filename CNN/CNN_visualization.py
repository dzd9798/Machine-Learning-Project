import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.models import load_model
from matplotlib import pyplot as plt
import cv2
import numpy as np

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():
    model = load_model('/home/dzd/dzd/labwork/face/CNN/model/CNN_model.h5')

    image = cv2.imread(os.path.join('/home/dzd/dzd/labwork/face/yaleBExtData/yaleB01', 'yaleB01_P00A-005E-10.pgm'), cv2.IMREAD_GRAYSCALE)
#    images=cv2.imread("/home/dzd/dzd/labwork/face/yaleBExtData/yaleB01/yaleB01_P00A-005E-10.pgm")
#    cv2.imshow("Image", images)
#    cv2.waitKey(0)

    # Turn the image into an array.
    # 根据载入的训练好的模型的配置，将图像统一尺寸
#    image_arr = cv2.resize(image, (192, 168))
    image.resize(192, 168)
    image_arr = np.array(image).reshape(1,192,168,1)
#    image_arr = np.expand_dims(image_arr, axis=0)

    # 第一个 model.layers[0],不修改,表示输入数据；
    # 第二个model.layers[ ],修改为需要输出的层数的编号[]
    layer_1 = K.function([model.layers[0].input], [model.layers[6].output])
#    visualization_model = K.function([model.layers[0].input], [model.layers[1].output])
    # 只修改inpu_image
#    f1 = visualization_model.predict(images/255.0)
    f1 = layer_1([image_arr/255.0])[0]

    # 第一层卷积后的特征图展示，输出是（1,66,66,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
    for _ in range(32):
                show_img = f1[:, :, :, _]
                show_img = show_img.reshape(len(show_img[0]),len(show_img[0][0]))
#                show_img.shape = [66, 66]
                plt.subplot(4, 8, _ + 1)
                # plt.imshow(show_img, cmap='black')
                plt.imshow(show_img, cmap='gray')
                plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
