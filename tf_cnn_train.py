import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tf_cnn_input_data
import tf_cnn_model

NUM_CLASSES = tf_cnn_model.NUM_CLASSES
IMAGE_SIZE = tf_cnn_model.IMAGE_SIZE
IMAGE_COLORS = tf_cnn_model.IMAGE_COLORS
IMAGE_PIXELS = tf_cnn_model.IMAGE_PIXELS

# If you encounter with an error, tensorflow.python.framework.errors.OutOfRangeError: Nan in summary histogram for: HistogramSummary,
# it is due to too large learning rate, for large batch size, need low learning rate.
# https://github.com/tensorflow/tensorflow/issues/307

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/tf_log', 'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_string('train', 'train.csv', 'File name of train data')
tf.app.flags.DEFINE_string('test', 'test.csv', 'File name of test data')
tf.app.flags.DEFINE_integer('max_steps', 1000, 'Number of batches to run')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate')
tf.app.flags.DEFINE_string('model', 'model.ckpt', 'File name of model data')
tf.app.flags.DEFINE_boolean('resume', False, 'Flag whether resuming or not')

train_csv = FLAGS.train
test_csv = FLAGS.test
print 'train data =', train_csv
print 'test data =', test_csv
print 'resume =', FLAGS.resume
print 'log directory=', FLAGS.train_dir

if IMAGE_COLORS == 1:
    dataSets = tf_cnn_input_data.DataSets(grayScale=True, train_data=train_csv, test_data=test_csv, shuffle=True)
else:
    dataSets = tf_cnn_input_data.DataSets(grayScale=False, train_data=train_csv, test_data=test_csv, shuffle=True)
print 'Reading datasets...'
input_data = dataSets.read_data_sets()
print 'done.'
train_image = input_data["train"]["images"]
train_label = input_data["train"]["labels"]
test_image = input_data["test"]["images"]
test_label = input_data["test"]["labels"]

# if IMAGE_COLORS == 1:
#     plt.imshow(train_image[0].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')  # height, width
# else:
#     plt.imshow(train_image[0].reshape(IMAGE_SIZE, IMAGE_SIZE, IMAGE_COLORS))  # height, width, channels
# plt.show()

with tf.Graph().as_default():
    x = tf.placeholder("float", shape=[None, IMAGE_PIXELS])
    y_ = tf.placeholder("float", shape=[None, NUM_CLASSES])
    keep_prob = tf.placeholder("float")

    logits = tf_cnn_model.inference(x, keep_prob)
    loss_value = tf_cnn_model.loss(logits, y_)
    train_op = tf_cnn_model.training(loss_value, FLAGS.learning_rate)
    acc = tf_cnn_model.accuracy(logits, y_)

    summary_op = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)

        sess.run(init_op)
        if FLAGS.resume and os.path.exists(FLAGS.model):
            saver.restore(sess, FLAGS.model)

        batch_size = FLAGS.batch_size
        start_batch = 0
        for i in range(FLAGS.max_steps):
            batch = [train_image[start_batch:start_batch + batch_size], train_label[start_batch:start_batch + batch_size]]
            start_batch += batch_size

            if start_batch > len(train_image) - batch_size:
                start_batch = 0
                perm = np.arange(len(train_image))
                np.random.shuffle(perm)
                train_image = train_image[perm]
                train_label = train_label[perm]

            if i % 100 == 0:
                train_accuracy = acc.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                writer.add_summary(summary_str, i)
                print "step %d, training accuracy %g" % (i, train_accuracy)
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model)
                saver.save(sess, checkpoint_path, global_step=i)
            train_op.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print "test accuracy %g " % acc.eval(feed_dict={x: test_image, y_: test_label, keep_prob: 1.0})
        saver.save(sess, FLAGS.model)
