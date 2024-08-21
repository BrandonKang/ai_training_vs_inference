from gluoncv.model_zoo import get_model
import matplotlib.pyplot as plt
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from PIL import Image
import io
import flask
import json

app = flask.Flask(__name__)

# Load the pre-trained model
model = get_model('cifar_resnet20_v1', classes=10, pretrained=True)

# CIFAR-10 class names list
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        if flask.request.files.get("img"):
            try:
                img = Image.open(io.BytesIO(flask.request.files["img"].read())).convert('RGB')

                transform_fn = transforms.Compose([
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])

                img = transform_fn(nd.array(img)).expand_dims(axis=0)

                # Perform prediction using the model
                pred = model(img)
                ind = nd.argmax(pred, axis=1).astype('int')

                # Return prediction result
                response = {
                    "prediction": class_names[ind.asscalar()],
                    "probability": float(nd.softmax(pred)[0][ind].asscalar())
                }

                return flask.jsonify(response)

            except Exception as e:
                return flask.jsonify({"error": str(e)})

    return flask.jsonify({"error": "No image provided or incorrect request method"})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
