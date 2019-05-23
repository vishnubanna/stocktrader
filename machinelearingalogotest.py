from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) #one_hot is a way of formatting data that makes it more readable to a computer (look into this) uses a binary format that makes the data more readable to the computer

#hyperparameter
#what we use to mess with to improve the algorithem
learning_rate = 0.001 #larger = faster, smaller = accurate
training_iters = 200000 #more itterations = more accurate
batch_size = 128 # how many examples, to calculate the loss over, smaller = more accurate, larger = more efficient
display_step = 10 # how often to print the cost/loss

#network parameters
n_input = 784 #image shape = 28x28, 784 pixels
n_classes = 10 # how many different labels, 10 digits 0-9, 10 labels
dropout = 0.75 # forces some nurons to turn off in attempt to force the algorithem to find new paths to the right answer. helps to prevent over fitting. so if dont use dropout, the model will be too fit to the data you tested on. randomly turns off nurons, so the data is forced to find new pathways. makes a more generalized model
# it is a probability value

#place holders
x = tf.placeholder(tf.float32, [None, n_input]) # gateway for the x vector inputs, or the image
y = tf.placeholder(tf.float32, [None, n_classes]) # gateway for the labels

keep_prob = tf.placeholder(tf.float32) #this is for the drop out to feed into the network

def conv2d(x, W, b, strides=1):
    #convolution = takes a part of the image and transforms it in some way
    #it tranforms the image in some way
    #the higher you go in the network the more abstract it gets
    #its like putting filters on an image
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'SAME')
    #strides are out tensors, tensors = data
    x = tf.nn.bias_add(x, b)
    #makes the model more accurate
    return tf.nn.relu(x)
    # returns a relu function: rectified linear unit: it is an activation function

def maxpool2d(x, k=2):
    # takes little segments of the image and returns the maximum of it
    return tf.nn.max_pool(x, ksize = [1,k,k,1], strides = [1,k,k,1], padding = 'SAME')

#create model
def conv_net(x, weights, biases, dropout):
    #reshape the input data to fit the model better
    x = tf.reshape(x, shape=[-1, 28, 28, 1]) #reshapes the input
    #--> this is the first layer
    #convolutional layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    #max_pooling layer
    conv1 = maxpool2d(conv1, k = 2)

    #--> create the next layer: takes the previous layer as inputs
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #then max pool this layer (breakdown and analyze in small chuncks)
    conv2 = maxpool2d(conv2, k = 2)

    #now make a fully connected layer
    #a fully connected layer is a generic layer, so every nuron in the fully conencted layer is connected to evey nuron in the previous layer
    # convolution layer breaks down the image to make reconitions
    # the fully connected layer does not, it just represents the data

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    #this is where the matirx multiplication happens, where the clasification occurs
    fc1 = tf.add(tf.matmul(fc1, weights['wd1'], biases['bd1']))
    fc1 = tf.nn.relu(fc1)

    #now apply the dropout
    fc1 = tr.nn.dropout(fc1, dropout)

    #output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out'], biases['out']))
    return out

#create weights
weights = {

    'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
    # 5 by 5 convolution, with 1 input(image) and 32 outputs(number of bits) 32 connections
    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
    # 5 by 5 convolution, with 32 input connections from previous and 64 outputs( connections in bits)
    'wd1': tf.Variable(tf.random_normal([7*7*64,1024])),
    # 7*7*64 inputs, 1024 output connections in bits
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
    # 1024 inputs, 10 outputs, your classes

}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'bc1': tf.Variable(tf.random_normal([n_classes])),
}

#construct model
prediction = conv_net(x, weights, biases, keep_prob)

#define optimizer and define loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    #measures the porbability of error in 2 mutually exclusive inputs

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    #AdamOptimizer uses gradient decent

#evaluate the model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#initalize the variables
init = tf.initialize_all_variables()

#launch all the graphs
with tf.Session as sess:
    sess.run(init)
    step = 1
    #keep running for the entirity of max itterations
    while(step*batch_size < training_iters):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout})
        print('iteration', step)

        if (step % display_step) == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y:batch_y, keep_prob: 1.})
            print("Iter {}" + str(step*batch_size) + ", Minibatch Loss = {:.6f}".format(loss) + ", Training accuracy = {:.6f}".format(acc))
        step += 1
    print("optimization finished")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
