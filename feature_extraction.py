import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
import cv2
import matplotlib.pyplot as plt
np.set_printoptions(threshold=1e20)
np.set_printoptions(linewidth=1e20)
img= cv2.imread(r'D:\pycharm_community\python_workspace\image_transformation/3.jpg')
img=cv2.resize(img.astype(np.float), (227, 227))
test_img= cv2.imread(r'D:\pycharm_community\python_workspace\similarity_calculate\alexnet/00001.jpg')
test_img=cv2.resize(test_img.astype(np.float), (227, 227))
x = tf.placeholder("float", [1, 227, 227, 3])
# conv1
weights1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                         stddev=1e-1), name='weights1')
biases1 = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                      trainable=True, name='biases1')
weights2 = tf.Variable(tf.truncated_normal([5, 5, 48, 256], dtype=tf.float32,
                                         stddev=1e-1), name='weights2')
biases2 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                     trainable=True, name='biases2')
weights3 = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
                                         dtype=tf.float32,
                                         stddev=1e-1), name='weights3')
biases3 = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                     trainable=True, name='biases3')
weights4 = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                         dtype=tf.float32,
                                         stddev=1e-1), name='weights4')
biases4 = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                     trainable=True, name='biases4')
weights5 = tf.Variable(tf.truncated_normal([3, 3, 192, 256],
                                         dtype=tf.float32,
                                         stddev=1e-1), name='weights5')
biases5 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                     trainable=True, name='biases5')
# weights6 = tf.Variable(tf.truncated_normal([6 * 6 * 256, 100],
#                                           dtype=tf.float32,
#                                           stddev=1e-1), name='weights6')
# biases6 = tf.Variable(tf.constant(0.0, shape=[100], dtype=tf.float32),
#                      trainable=True, name='biases6')
#
# weights7 = tf.Variable(tf.truncated_normal([100, 1],
#                                           dtype=tf.float32,
#                                           stddev=1e-1), name='weights7')
# biases7 = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
#                      trainable=True, name='biases7')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
weights_dict = np.load(os.path.join(r'D:\pycharm_community\python_workspace\similarity_calculate/', 'bvlc_alexnet.npy'),allow_pickle=True, encoding='latin1').item()
for op_name in weights_dict:
    if op_name == 'conv1':
        for data in weights_dict[op_name]:
            # Biases
            if len(data.shape) == 1:
                sess.run(tf.assign(biases1,data))
            # Weights
            else:
                sess.run(tf.assign(weights1,data))
    elif op_name == 'conv2':
        for data in weights_dict[op_name]:
            # Biases
            if len(data.shape) == 1:
                sess.run(tf.assign(biases2,data))
            # Weights
            else:
                sess.run(tf.assign(weights2,data))
    elif op_name == 'conv3':
        for data in weights_dict[op_name]:
            # Biases
            if len(data.shape) == 1:
                sess.run(tf.assign(biases3,data))
            # Weights
            else:
                sess.run(tf.assign(weights3,data))
    elif op_name == 'conv4':
        for data in weights_dict[op_name]:
            # Biases
            if len(data.shape) == 1:
                sess.run(tf.assign(biases4,data))
            # Weights
            else:
                sess.run(tf.assign(weights4,data))
    elif op_name == 'conv5':
        for data in weights_dict[op_name]:
            # Biases
            if len(data.shape) == 1:
                sess.run(tf.assign(biases5,data))
            # Weights
            else:
                sess.run(tf.assign(weights5,data))
    # if op_name == 'fc6':
    #     for data in weights_dict[op_name]:
    #         # Biases
    #         if len(data.shape) == 1:
    #             sess.run(tf.assign(biases6,data))
    #         # Weights
    #         else:
    #             sess.run(tf.assign(weights6,data))
    # elif op_name == 'fc7':
    #     for data in weights_dict[op_name]:
    #         # Biases
    #         if len(data.shape) == 1:
    #             sess.run(tf.assign(biases7,data))
    #         # Weights
    #         else:
    #             sess.run(tf.assign(weights7,data))



#sess.run(x,feed_dict={x:img.reshape((1, 227, 227, 3))})


conv1 = tf.nn.conv2d(x, weights1, [1, 4, 4, 1], padding='SAME')
bias1 = tf.nn.bias_add(conv1, biases1)
conv11 = tf.nn.relu(bias1, name='conv1')

# lrn1
lrn1 = tf.nn.local_response_normalization(conv11,
                                          alpha=1e-4,
                                          beta=0.75,
                                          depth_radius=2,
                                          bias=2.0)

# pool1
pool1 = tf.nn.max_pool(lrn1,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID')

# conv2
pool1_groups = tf.split(axis=3, value=pool1, num_or_size_splits=2)

kernel_groups = tf.split(axis=3, value=weights2, num_or_size_splits=2)
conv_up = tf.nn.conv2d(pool1_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
conv_down = tf.nn.conv2d(pool1_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
biases_groups = tf.split(axis=0, value=biases2, num_or_size_splits=2)
bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
bias2 = tf.concat(axis=3, values=[bias_up, bias_down])
conv22 = tf.nn.relu(bias2, name='conv2')

# lrn2
lrn2 = tf.nn.local_response_normalization(conv22,
                                          alpha=1e-4,
                                          beta=0.75,
                                          depth_radius=2,
                                          bias=2.0)

# pool2
pool2 = tf.nn.max_pool(lrn2,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID')

# conv3
conv3 = tf.nn.conv2d(pool2, weights3, [1, 1, 1, 1], padding='SAME')
bias3 = tf.nn.bias_add(conv3, biases3)
conv33 = tf.nn.relu(bias3, name='conv3')

# conv4
conv3_groups = tf.split(axis=3, value=conv33, num_or_size_splits=2)
kernel_groups = tf.split(axis=3, value=weights4, num_or_size_splits=2)
conv_up = tf.nn.conv2d(conv3_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
conv_down = tf.nn.conv2d(conv3_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
biases_groups = tf.split(axis=0, value=biases4, num_or_size_splits=2)
bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
bias4 = tf.concat(axis=3, values=[bias_up, bias_down])
conv44 = tf.nn.relu(bias4, name='conv4')

# conv5
conv4_groups = tf.split(axis=3, value=conv44, num_or_size_splits=2)
kernel_groups = tf.split(axis=3, value=weights5, num_or_size_splits=2)
conv_up = tf.nn.conv2d(conv4_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
conv_down = tf.nn.conv2d(conv4_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
biases_groups = tf.split(axis=0, value=biases5, num_or_size_splits=2)
bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
bias5 = tf.concat(axis=3, values=[bias_up, bias_down])
conv55 = tf.nn.relu(bias5, name='conv5')

# pool5
pool5 = tf.nn.max_pool(conv55,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID', )


#flattened = tf.reshape(pool5, shape=[-1, 6 * 6 * 256])
flattened = tf.reshape(pool5, shape=[-1, 9216])

# # fc6
# bias6 = tf.nn.xw_plus_b(flattened, weights6, biases6)
# fc6 = tf.nn.relu(bias6)
#
# # dropout6
# dropout6 = tf.nn.dropout(fc6, keep_prob=0.5)
#
# # fc7
# bias = tf.nn.xw_plus_b(dropout6, weights7, biases7)
# fc7 = tf.nn.relu(bias)

conv_out1=sess.run(conv33,feed_dict={x:img.reshape((1, 227, 227, 3))})
conv_out2=sess.run(conv44,feed_dict={x:img.reshape((1, 227, 227, 3))})
conv_out3=sess.run(conv55,feed_dict={x:img.reshape((1, 227, 227, 3))})

conv_weights=sess.run(weights1)

# for i in range(96):
#     plt.subplot(8,12, i + 1)
#     plt.imshow(conv_out[0][:,:,i])
#     frame = plt.gca()
#     # y 轴不可见
#     frame.axes.get_yaxis().set_visible(False)
#     # x 轴不可见
#     frame.axes.get_xaxis().set_visible(False)
# plt.show()

##conv2_out
# for i in range(256):
#     plt.subplot(16,16, i + 1)
#     plt.imshow(conv_out[0][:,:,i])
#     frame = plt.gca()
#     # y 轴不可见
#     frame.axes.get_yaxis().set_visible(False)
#     # x 轴不可见
#     frame.axes.get_xaxis().set_visible(False)
# plt.show()
#conv3_out
for i in range(384):
    plt.subplot(16,24, i + 1)
    plt.imshow(conv_out1[0][:,:,i])
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
plt.show()
#conv4_out
for i in range(384):
    plt.subplot(16,24, i + 1)
    plt.imshow(conv_out2[0][:,:,i])
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
plt.show()
#conv5_out
for i in range(256):
    plt.subplot(16,16, i + 1)
    plt.imshow(conv_out3[0][:,:,i])
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
plt.show()
#print(conv_out1.shape,conv_out2.shape,conv_out3.shape)
# print(conv_weights.shape)
#for i in range(96):
#   fig1 = plt.figure(1)
#   conv1_weights=conv_weights[:,:, :, i]
#   plt.imshow(conv1_weights)
#   plt.savefig('conv1_weights/conv1_weights'+str(i)+'.png',dpi=600)
#   plt.show()

