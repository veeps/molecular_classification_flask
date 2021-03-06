# imports
import numpy as np
import os
from flask import Flask, request, render_template, jsonify, redirect, json
from werkzeug.utils import secure_filename
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

                with open("assets/model.json") as f:
                    data = json.loads(f.read())
                    data = json.dumps(data) # converts json file into string

                model = model_from_json(data)

                test_image = image.load_img(img_path, target_size=(150, 150))
                img_tensor = image.img_to_array(test_image)                    # (height, width, channels)
                img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
                img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

                prediction = model.predict_classes(img_tensor)
                if prediction[0] == 0:
                    prediction = "antineoplastic"
                elif prediction[0] == 1:
                    prediction = "CNS"
                elif prediction[0] == 2:
                    prediction = "cardio"

                return render_template("/results.html", prediction=prediction, imgpath = img_path)

            else:
                print("That file extension is not allowed")
                return redirect(request.url)

    return render_template("/upload_image.html")




# run the app
if __name__ == '__main__':
    app.run(debug=True)
