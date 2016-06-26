import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
#IMAGE_COLORS = 3
IMAGE_COLORS = 1
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * IMAGE_COLORS


def inference(images, keep_prob):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_COLORS])
    tf.image_summary('images', x_image)

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, IMAGE_COLORS, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        tf.histogram_summary('W_conv1', W_conv1)
        tf.histogram_summary('b_conv1', b_conv1)
        tf.histogram_summary('h_conv1', h_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        tf.histogram_summary('W_fc2', W_fc2)
        tf.histogram_summary('b_fc2', b_fc2)

    return y_conv


def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels * tf.log(logits))
    tf.scalar_summary('cross entropy', cross_entropy)
    loss = tf.reduce_mean(cross_entropy)
    return loss


def training(loss, learning_rate):
    tf.scalar_summary("loss", loss)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary('accuracy', acc)
    return acc
