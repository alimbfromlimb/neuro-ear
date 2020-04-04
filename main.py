from launch_ import *
from flask import *

import os
import sys
import librosa
from keras.models import load_model

app = Flask(__name__)

model = load_model('my_model_one_sec_19.h5')
model._make_predict_function()

print(sys.version)


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/classify")
def classify():
    return render_template("classify.html")


# @app.route('/upload')
# def upload():
#     return render_template("file_upload_form.html")


# @app.route("/aboutUs")
# def about():
#     return render_template("aboutUs.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        track_name = f.filename
        assert track_name[-3:] == 'wav', track_name
        f.save(track_name)
        print(track_name)

        # track_librosa, sr = librosa.load(track_name, duration=20.0)
        # os.remove(track_name)
        # librosa.output.write_wav(track_name, track_librosa, sr)

        inst = instrument_classifier(track_name, model)
        print(inst)
        os.remove(track_name)
        return render_template("success__"+str(inst)+".html", name=f.filename)


if __name__ == '__main__':
    app.run(debug=True)