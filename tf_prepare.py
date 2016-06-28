import skimage.io
from skimage.transform import resize
import tensorflow as tf
import numpy as np

image = skimage.io.imread('obama.jpg') / 255.0
image = resize(image, (224, 224))
image = image.reshape((1, 224, 224, 3))
content = tf.Variable(image, dtype=tf.float32)

with open("vggface16.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

tf.import_graph_def(graph_def, input_map={"images": content})

graph = tf.get_default_graph()

fc8 = graph.get_tensor_by_name("import/prob:0")
fc7 = graph.get_tensor_by_name("import/Relu_1:0")
conv5_3 = graph.get_tensor_by_name("import/conv5_3/Relu:0")

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    fc8_, fc7_, conv5_3_ = sess.run([fc8, fc7, conv5_3])
    np.save('obama_fc8.npy', fc8_)
    np.save('obama_fc7.npy', fc7_)
    np.save('obama_conv5_3.npy', conv5_3_)
