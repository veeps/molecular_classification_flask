# imports
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json

# initialize the flask app
app = Flask('myApp')

# save images
app.config["IMAGE_UPLOADS"] = "static/images"

### Upload image
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        if request.files:

            image_user = request.files["image"]

            if image_user.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_image(image_user.filename):
                filename = secure_filename(image_user.filename)

                image_user.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                img_path = "static/images/"+filename


                json_string = '{"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": [null, 150, 150, 3], "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow"}'

                model = model_from_json(json_string)

                test_image = image.load_img(img_path, target_size=(150, 150))
                img_tensor = image.img_to_array(test_image)                    # (height, width, channels)
                img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
                img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

                prediction = model.predict_classes(img_tensor)
                if prediction[0] == 0:
                    prediction = "antineoplastic"
                elif prediction[0] == 1:
                    prediction = "CNS"
                elif prediction[0] == 1:
                    prediction = "cardio"

                return render_template("/results.html", prediction=prediction, imgpath = img_path)

            else:
                print("That file extension is not allowed")
                return redirect(request.url)

    return render_template("/upload_image.html")




# run the app
if __name__ == '__main__':
    app.run(debug=True)
