from launch_ import *
from flask import *
# from pydub import AudioSegment

import os


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


@app.route('/upload')
def upload():
    return render_template("file_upload_form.html")


@app.route("/aboutUs")
def about():
    return render_template("aboutUs.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        track_name = f.filename
        f.save(track_name)
        assert track_name[-3:] == 'wav' or track_name[-3:] == 'mp3', track_name
        # if track_name[-3:] == 'mp3':
        #     sound = AudioSegment.from_mp3(track_name)
        #     os.remove(track_name)
        #     sound.export("file.wav", format="wav")
        #     track_name = "file.wav"
        print(f.filename)
        inst = instrument_classifier(track_name, model)
        print(inst)
        os.remove(f.filename)
        return render_template("success__"+str(inst)+".html", name=f.filename)


if __name__ == '__main__':
    app.run(debug=True)