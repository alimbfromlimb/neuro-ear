from launch_ import *
from flask import *

import os

app = Flask(__name__)

model = load_model('my_model_one_sec_19.h5')
model._make_predict_function()


@app.route('/')
def home():
    return render_template("home.html")


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
        f.save(f.filename)
        track_name = f.filename
        print(f.filename)
        inst = instrument_classifier(track_name, model)
        print(inst)
        os.remove(f.filename)
        return render_template("success_"+str(inst)+".html", name=f.filename)


if __name__ == '__main__':
    app.run(debug=True)