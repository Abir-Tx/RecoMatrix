# To test that on which device the code is running
import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

c = a + b

print("Device of operation 'c':", c.device)
