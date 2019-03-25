from flask import Flask, session, redirect, url_for, render_template, abort, request, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
from gan import gan


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/upload'
app.secret_key = 'this_is_a_secret_key'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/models/<model_name>', methods=['GET', 'POST'])
def models(model_name):
    model_names = {'sketch2faces': 'Sketch to Faces', 'maps2buildings': 'Maps to Buildings', 'facades2buildings': 'Facades to Buildings', 'scapes2city': 'Cityscapes to Streets'}
    if request.method == 'POST':
        gan.generator.load_weights(f'models/{model_name}_generator.h5')
        gan.discriminator.load_weights(f'models/{model_name}_discriminator.h5')
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.save(os.path.join(os.getcwd() + '/' + app.config['UPLOAD_FOLDER'], 'sample.jpg'))
            gan.sample_images()
            return render_template('generator_page.html', model_name=model_names[model_name], images=True, generated_image='/generated/sample.png', original_image='/upload/sample.jpg')
    else:
        return render_template('generator_page.html', model_name=model_names[model_name], images=False, generated_image=None, original_image=None)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/generated/<path:path>')
def generated(path):
    return send_from_directory('generated', path)

@app.route('/upload/<path:path>')
def uploaded(path):
    return send_from_directory('upload', path)

@app.route('/static/<path:path>')
def staticpath(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run()
