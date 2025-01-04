from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the model and kernel at startup
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]


# Route to render the homepage (form to upload image)
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle the image upload and processing
@app.route('/colorize', methods=['POST'])
def colorize():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    
    # Save the uploaded image
    upload_path = os.path.join('static/uploads', file.filename)
    file.save(upload_path)
    
    # Process the uploaded image
    bw_image = cv2.imread(upload_path)
    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")

    # Save the colorized image
    colorized_image_path = os.path.join('static/uploads', 'colorized_' + file.filename)
    cv2.imwrite(colorized_image_path, colorized)

    return render_template('index.html', original_image=file.filename, colorized_image='colorized_' + file.filename)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    app.run(host = "0.0.0.0", port=port)
