import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image,ImageChops,ImageEnhance
import numpy as np
import os
import cv2
rootdir=r'D:\matlab2018b\matlab_workspace\fault_detection\aa'
path=os.listdir(rootdir)
for file in path:
    img_path=os.path.join(rootdir,file)
    #origin_img=Image.open(img_path)
    origin_img = cv2.imread(img_path)
    #rotation_range
    # img = tf.image.rot90(origin_img, 1)
    # with tf.Session() as sess:
    #     img = sess.run(img)
    #     plt.subplot(1,2,1)
    #     plt.title("tensorflow")
    #     plt.imshow(img)

    # PIL
    #img = origin_img.rotate(90)  # 逆时针旋转
    # plt.imshow(img)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    #img.save(r'D:\matlab2018b\matlab_workspace\fault_detection\aa/'+file.split('.')[0]+'.jpg')
    # #plt.subplot(1, 2, 2)
    # plt.title("rotation_range")
    # plt.imshow(img)
    # plt.show()

    #horiaontal_flip
    with tf.Session() as sess:
    # # #     # tensorflow
    #       flipped1 = tf.image.flip_left_right(origin_img)
    #       flipped1 = flipped1.eval()
    #       cv2.imwrite(r'D:\matlab2018b\matlab_workspace\fault_detection\aughole/' + file.split('.')[0] + '.jpg',flipped1)
    #       flipped2 = tf.image.flip_up_down(origin_img)
    #       flipped2 = flipped2.eval()
    #       cv2.imwrite(r'D:\matlab2018b\matlab_workspace\fault_detection\aughole/' + file.split('.')[0] + '.jpg',flipped2)
          flipped3 = tf.image.transpose_image(origin_img)
          flipped3 = flipped3.eval()
          cv2.imwrite(r'D:\matlab2018b\matlab_workspace\fault_detection\aughole/' + file.split('.')[0] + '.jpg',flipped3)
        # transpose_img = tf.image.transpose_image(origin_img)
    # PIL
    #flipped1=Image.FLIP_LEFT_RIGHT(origin_img)
    #flipped1.save(r'D:\matlab2018b\matlab_workspace\trans_fault_detection\horizontal_flip/' + file.split('.')[0] + '.png')
    #flipped2 = Image.FLIP_UP_DOWN(origin_img)
    #transpose_img = Image.TRANSPOSE(origin_img)
        # plt.title("horizontal_flip")
        # plt.imshow(flipped1.eval())
        # plt.show()


# for i, img in enumerate([origin_img, flipped1, flipped2, transpose_img]):
#     plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     plt.imshow(img.eval())
# plt.show()


    #zoom_rang
    # origin_size=origin_img.size
    # img = origin_img.resize((origin_size[0]//3,origin_size[1]//3),resample=Image.LANCZOS)
    # plt.title("zoom_range(1/3)")
    # plt.imshow(img)
    # plt.show()

#shear_range
# img = origin_img.crop((100, 300, 1500, 1500))  # 参数为坐标左上右下
# plt.imshow(img)
# plt.title("shear_range")
# plt.show()

#width_shift/height_shift
# img = ImageChops.offset(origin_img, 100, 50)
# width, height =img.size
# img.paste((0, 0, 0), (0, 0, 100, height))
# img.paste((0, 0, 0), (0, 0, width, 50))
# plt.title("shift")
# plt.imshow(img)
# plt.show()



#fill_mode