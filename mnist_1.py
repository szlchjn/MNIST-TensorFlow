import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.global_variables_initializer()

XX = tf.reshape(X, [-1, 28*28])
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))

correct_prediction= tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.5)

train_step = optimizer.minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(init)

img_test, lab_test = mnist.test.images, mnist.test.labels
test_data = {X: img_test, Y_: lab_test}

iterations = 5000
max_a = 0
training_acc = {}
testing_acc = {}
training_loss = {}
testing_loss = {}

for i in range(iterations+1):

    img_train, lab_train = mnist.train.next_batch(100)
    train_data = {X: img_train, Y_: lab_train}
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
plt.ylim((0, 0.4))
plt.plot([key for key in training_loss.keys()], [val for val in training_loss.values()], color='blue', linewidth=1)
plt.plot([key for key in testing_loss.keys()], [val for val in testing_loss.values()], color='red', linewidth=2)
plt.show()

