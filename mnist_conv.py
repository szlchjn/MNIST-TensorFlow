import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False, validation_size=0)
tf.set_random_seed(0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1))
b1 = tf.Variable(tf.ones([4])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, 4, 8], stddev=0.1))
b2 = tf.Variable(tf.ones([8])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, 8, 12], stddev=0.1))
b3 = tf.Variable(tf.ones([12])/10)

W4 = tf.Variable(tf.truncated_normal([7*7*12, 200], stddev=0.1))
b4 = tf.Variable(tf.ones([200])/10)
W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
b5 = tf.Variable(tf.ones([10])/10)

stride = 1
Y1conv = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1 = tf.nn.relu(Y1conv + b1)
stride = 2
Y2conv = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2 = tf.nn.relu(Y2conv + b2)
stride = 2
Y3conv = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3 = tf.nn.relu(Y3conv + b3)

YY = tf.reshape(Y3, [-1, 7*7*12])
Y4 = tf.nn.relu(tf.matmul(YY, W4) + b4)

Ylogits = tf.matmul(Y4, W5) + b5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

correct_prediction= tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(lr)

train_step = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

img_test, lab_test = mnist.test.images, mnist.test.labels
test_data = {X: img_test, Y_: lab_test}

iterations = 10000
max_a = 0
training_acc = {}
testing_acc = {}
training_loss = {}
testing_loss = {}

for i in range(iterations+1):
    max_lr = 0.003
    min_lr = 0.0001
    decay = 2000.0
    learning_rate = min_lr + (max_lr - min_lr) * np.exp(-i/decay)

    img_train, lab_train = mnist.train.next_batch(100)
    train_data = {X: img_train, Y_: lab_train, lr: learning_rate}
    sess.run(train_step, feed_dict=train_data)
    if i % 20 == 0:
        print('Training:', sess.run([accuracy, cross_entropy], feed_dict=train_data))
        a_tr, c_tr = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        training_acc[i] = a_tr
        training_loss[i] = c_tr
    if i % 100 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print('\nIteration: {2}\nTest accuracy: {0}\nTest loss: {1}'.format(a, c, i))
        testing_acc[i] = a
        testing_loss[i] = c

        if a > max_a:
            max_a = a
            print('\nMax accuracy: {}\n'.format(max_a))

plt.figure(1)
plt.subplot(121)
plt.title('Accuracy')
plt.grid(True)
plt.ylim((0.9, 1))
plt.plot([key for key in training_acc.keys()], [val for val in training_acc.values()], color='blue', linewidth=1)
plt.plot([key for key in testing_acc.keys()], [val for val in testing_acc.values()], color='red', linewidth=2)

plt.subplot(122)
plt.title('Loss')
plt.grid(True)
plt.ylim((0, 0.3))
plt.plot([key for key in training_loss.keys()], [val for val in training_loss.values()], color='blue', linewidth=1)
plt.plot([key for key in testing_loss.keys()], [val for val in testing_loss.values()], color='red', linewidth=2)
plt.show()

