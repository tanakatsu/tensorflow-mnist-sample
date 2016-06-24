import os
import tensorflow as tf
import multiprocessing as mp
from flask import Flask, render_template, redirect, url_for, request
from werkzeug import secure_filename


ALLOWED_EXTENSIONS = set(['jpg'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def heroku_env():
    if 'DYNO' in os.environ:
        return True
    else:
        return False

if heroku_env():
    UPLOAD_FOLDER = '/tmp'
else:
    UPLOAD_FOLDER = '.'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return redirect(url_for('upload'))


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
            return redirect(url_for('hello_world'))
        if f:
            return '''
            <h3>Not supported file</h3>
            '''
        else:
            return '''
            <h3>file is not attached</h3>
            '''

if __name__ == '__main__':
    app.debug = True
    app.run()
