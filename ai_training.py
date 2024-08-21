from gluoncv.model_zoo import get_model
from mxnet import gluon, autograd, nd, init, context
from mxnet.gluon.data.vision import transforms as mx_transforms
from mxnet.gluon.data.vision import CIFAR10
import flask
import json
from PIL import Image
import io

app = flask.Flask(__name__)

# Set the context to be used (CPU or GPU).
ctx = context.cpu()

# Prepare the dataset and create the data loader.
transform_train = mx_transforms.Compose([
    mx_transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Function that can replace RandomCrop
    mx_transforms.RandomFlipLeftRight(),  # Use RandomFlipLeftRight from MXNet
    mx_transforms.ToTensor(),
    mx_transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = mx_transforms.Compose([
    mx_transforms.ToTensor(),
    mx_transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

train_data = gluon.data.DataLoader(
    CIFAR10(train=True).transform_first(transform_train), batch_size=128, shuffle=True, last_batch='discard')

test_data = gluon.data.DataLoader(
    CIFAR10(train=False).transform_first(transform_test), batch_size=128, shuffle=False, last_batch='discard')

# Model definition and initialization
model = get_model('cifar_resnet20_v1', classes=10)
model.initialize(init.Xavier(), ctx=ctx)

# Set loss function and optimizer
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.1, 'momentum': 0.9, 'wd': 0.0001})

# Model training function
def train_model(epochs):
    for epoch in range(epochs):
        train_loss = 0.0
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = model(data)
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(batch_size=128)
            train_loss += loss.mean().asscalar()
        print(f"Epoch {epoch+1}, Loss: {train_loss/len(train_data):.3f}")

# Train the model
train_model(epochs=10)

# Save model parameters
model.save_parameters('cifar_resnet20_v1.params')

# Load pre-trained model parameters
model.load_parameters('cifar_resnet20_v1.params', ctx=ctx)

# CIFAR-10 class names list
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        if flask.request.files.get("img"):
            try:
                img = Image.open(io.BytesIO(flask.request.files["img"].read())).convert('RGB')

                transform_fn = mx_transforms.Compose([
                    mx_transforms.Resize(32),
                    mx_transforms.CenterCrop(32),
                    mx_transforms.ToTensor(),
                    mx_transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])

                img = transform_fn(nd.array(img)).expand_dims(axis=0).as_in_context(ctx)

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
