import skimage.io
from skimage.transform import resize
import tensorflow as tf
import numpy as np
import glob
import sys
from matplotlib import pyplot as plt
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops, dtypes

global_step = tf.Variable(0, name='global_step', trainable=False)

guide_image = skimage.io.imread('obama.jpg') / 255.0
guide_image = resize(guide_image, (224, 224))
guide_image = guide_image.reshape((1, 224, 224, 3))

source_image = skimage.io.imread('face.jpg') / 255.0
source_image = resize(source_image, (224, 224))
source_image = source_image.reshape((1, 224, 224, 3))

# noise_image = skimage.io.imread('30000.jpg') / 255.0
# noise_image = resize(noise_image, (224, 224))
# noise_image = noise_image.reshape((1, 224, 224, 3))
# content = tf.Variable(noise_image, dtype=tf.float32)

content = tf.Variable(source_image, dtype=tf.float32)


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img

with open("vggface16.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

tf.import_graph_def(graph_def, input_map={"images": content})

graph = tf.get_default_graph()

fc8 = graph.get_tensor_by_name("import/prob:0")
fc7 = graph.get_tensor_by_name("import/Relu_1:0")

guide_fc8 = np.load('obama_fc8.npy')
guide_fc7 = np.load('obama_fc7.npy')

guide_fc8 = tf.reshape(guide_fc8, (-1, 2622))
fc8 = tf.reshape(fc8, (-1, 2622))

guide_fc7 = tf.reshape(guide_fc7, (-1, 4096))
fc7 = tf.reshape(fc7, (-1, 4096))

tv_loss = 2 * (
    (tf.nn.l2_loss(content[:, 1:, :, :] - content[:, :224 - 1, :, :]) /
     224 * 223 * 3) +
    (tf.nn.l2_loss(content[:, :, 1:, :] - content[:, :, :224 - 1, :]) /
     223 * 224 * 3))

fc8_loss = 10e5 * tf.reduce_sum(tf.square(tf.sub(guide_fc8, fc8)))
fc7_loss = tf.reduce_sum(tf.square(tf.sub(guide_fc7, fc7)))

loss = fc8_loss + fc7_loss + tv_loss

starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           500, 0.99, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate)
opt = optimizer.minimize(
    loss, global_step=global_step, gate_gradients=optimizer.GATE_NONE)


with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    # for op in sess.graph.get_operations():
    #     print op.name
    # sys.exit()

    for i in range(20000):
        opt_, content_, loss_, fc8_loss_, tv_loss_, fc7_loss_ = sess.run(
            [opt, content, loss, fc8_loss, tv_loss, fc7_loss])

        step = sess.run(global_step)

        print {"loss_": loss_, "fc8_loss_": fc8_loss_, "tv_loss_": tv_loss_, "step": step, "fc7_loss": fc7_loss_}
        sys.stdout.flush()

        if step % 500 == 0:
            summary_image = concat_images(guide_image[0], content_[0])
            summary_image = concat_images(summary_image, source_image[0])
            plt.imsave('checkpoint_' + str(step), summary_image)
