import os
from flask import Flask, request, redirect, url_for, flash, render_template, send_from_directory
from werkzeug.utils import secure_filename
from scipy import misc
import tryReco
from time import clock

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ALLOWED_EXTENSIONS = set(['jpg'])

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

pnet, rnet, onet = tryReco.initMTCNN()

model_dir = './FaceNetModel'
mysess, images_placeholder, embeddings, phase_train_placeholder = tryReco.initFaceNet(model_dir)

idimg, id_name, ids = tryReco.get_idimg('./id')
feed_dict = {images_placeholder: idimg, phase_train_placeholder: False}
idemb = mysess.run(embeddings, feed_dict=feed_dict)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def show_name(name):
    return "Name: %s" % name

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            r = request.form.get("Recognize", "")
            s = request.form.get("SendID", "")
            print 'a'

            if r == "Recognize":
                print "yes"
                filename = file.filename

                target = os.path.join(APP_ROOT, 'images/')
                destination = "/".join([target, filename])
                file.save(destination)

                detectstart = clock()
                img, face_in_test, detected_img, bb = tryReco.align_data(destination, 160, 16, pnet, rnet, onet)
                detectedend = clock()
                print ("detect face time cost: %1.4f" % (detectedend - detectstart))

                filename_l = len(filename)
                detected_filename = filename[0:filename_l - 4] + '_detected.jpg'

                if (face_in_test > 0 and ids > 0):
                    feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                    reco_start = clock()
                    testemb = mysess.run(embeddings, feed_dict=feed_dict)
                    reco_end = clock()
                    print ("recognize time cost: %1.4f" % (reco_end - reco_start))

                    detected_img = tryReco.recognize(testemb, idemb, face_in_test, ids, detected_img, bb, id_name)
                    misc.imsave('./images/' + detected_filename, detected_img)

                else:
                    misc.imsave('./images/' + detected_filename, detected_img)
                    print ('no face detect.')

                allend = clock()
                print ("total time cost: %1.4f" % (allend - detectstart))
                return render_template("display_image.html", image_name=detected_filename)

            if s == "SendID":
                filename = file.filename

                target = os.path.join(APP_ROOT, 'tmpid/')
                destination = "/".join([target, filename])
                file.save(destination)
                print destination

                img, face_in_test, detected_img, _ = tryReco.align_data(destination, 160, 32, pnet, rnet, onet)

                if face_in_test == 1:
                    ID_name = './id/'+filename
                    misc.imsave(ID_name, img[0])
                    update_face()

                    filename_l = len(filename)
                    ID_filename = filename[0:filename_l - 4] + '_ID.jpg'
                    misc.imsave('./images/' +ID_filename,img[0])

                    return render_template("Send_ID_Success.html", image_name=ID_filename)
                else:
                    filename_l = len(filename)
                    detected_filename = filename[0:filename_l - 4] + '_detected.jpg'
                    misc.imsave('./images/' + detected_filename, detected_img)

                    return render_template("Send_ID_Fail.html", image_name=detected_filename)

    return render_template("css_index.html")

@app.route('/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

def update_face():
    global idimg, id_name, ids
    idimg, id_name, ids = tryReco.get_idimg('./id')
    feed_dict = {images_placeholder: idimg, phase_train_placeholder: False}
    global idemb
    idemb = mysess.run(embeddings, feed_dict=feed_dict)
    print 'update'


if __name__ == "__main__":
    app.run(host='0.0.0.0')
