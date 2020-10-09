from launch import *
from flask import *
from training.architecture import ConvNet
from torch import load
import os
import sys


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

net = ConvNet()
net.load_state_dict(load('adam_1.ckpt', map_location='cpu'))

print(sys.version)


@app.route('/')
def home():
    return render_template("new_index.html")


@app.route("/classify")
def classify():
    return render_template("new_classify.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        if os.path.exists("static/probs.png"):
            os.remove("static/probs.png")
        else:
            print("The file does not exist")
        f = request.files['file']
        track_name = f.filename
        assert track_name[-3:] == 'wav', track_name
        f.save(track_name)
        print(track_name)

        classifier_torch(track_name, net, 26)
        if os.path.exists(track_name):
            os.remove(track_name)
        else:
            print("The file does not exist")
        os.chmod("static/probs.png", 777)
        return render_template("new_success_graph.html", name=f.filename)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response


if __name__ == '__main__':
    # app.run(debug=True)
    # app.run(host='0.0.0.0')
    app.run()
