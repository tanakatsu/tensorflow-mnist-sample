import os
import tensorflow as tf
import multiprocessing as mp
from flask import Flask, render_template, redirect, url_for, request, send_from_directory
import numpy as np
import base64
import re
import tf_cnn_model
# from werkzeug import secure_filename

ALLOWED_EXTENSIONS = set(['jpg'])
NUM_CLASSES = tf_cnn_model.NUM_CLASSES
IMAGE_COLORS = tf_cnn_model.IMAGE_COLORS
IMAGE_SIZE = tf_cnn_model.IMAGE_SIZE
IMAGE_PIXELS = tf_cnn_model.IMAGE_PIXELS


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def heroku_env():
    if 'DYNO' in os.environ:
        return True
    else:
        return False


def load_image(filename):
    jpeg = tf.read_file(filename)
    if IMAGE_COLORS == 1:
        image = tf.image.decode_jpeg(jpeg, channels=1)
    else:
        image = tf.image.decode_jpeg(jpeg, channels=3)
    resized_image = tf.image.resize_images(image, IMAGE_SIZE, IMAGE_SIZE)
    return resized_image

if heroku_env():
    UPLOAD_FOLDER = '/tmp'
else:
    UPLOAD_FOLDER = '.'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

x = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
keep_prob = tf.placeholder("float")
logits = tf_cnn_model.inference(x, keep_prob)
fname = tf.placeholder("string")
resized_image = load_image(fname)
upload_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'tmp.jpg')
sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
saver.restore(sess, os.path.join(APP_ROOT, "model.ckpt"))


# http://stackoverflow.com/questions/13768007/browser-caching-issues-in-flask
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.route('/')
def index():
    # return redirect(url_for('upload'))
    return redirect(url_for('upload2'))  # hand writing


@app.route('/hello')
def hello_world():
    core_num = mp.cpu_count()
    config = tf.ConfigProto(
        inter_op_parallelism_threads=core_num,
        intra_op_parallelism_threads=core_num)
    sess = tf.Session(config=config)

    hello = tf.constant('Hello, TensorFlow!')
    return sess.run(hello)


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
            filename = 'tmp.jpg'
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('predict'))
        if f:
            return '''
            <h3>Not supported file</h3>
            '''
        else:
            return '''
            <h3>file is not attached</h3>
            '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict')
def predict():
    # Do not use .eval(). It does not work.
    resized_img = sess.run(resized_image, feed_dict={fname: upload_filename})
    logits_value = sess.run(logits, feed_dict={x: resized_img.reshape(-1, IMAGE_PIXELS) / 255.0, keep_prob: 1.0})[0]
    label = np.argmax(logits_value)
    score = logits_value[label]
    return render_template('predict.html', filename='tmp.jpg', label=label, score=score)


@app.route('/upload2')
def upload2():
        return render_template('upload2.html')


@app.route('/uploader2', methods=['POST'])
def uploader2():
    if request.method == 'POST':
        image_enc = request.form['image_base64']
        image_str = re.sub(r'^data:.+;base64,', '', image_enc)
        image_dec = base64.b64decode(image_str)
        with open('tmp.jpg', 'wb') as f:
            f.write(image_dec)
            return redirect(url_for('predict'))

        return '''
        <h3>image file is not attached</h3>
        '''

if __name__ == '__main__':
    app.debug = True
    app.run()
