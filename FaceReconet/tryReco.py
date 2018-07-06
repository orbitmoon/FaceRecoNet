from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import clock

import cv2
import math

def initMTCNN():
    MTCNNsess = tf.Session()
    print ('init MTCNN')
    start = clock()
    pnet, rnet, onet = align.detect_face.create_mtcnn(MTCNNsess, None)
    stop = clock()
    print ('load MTCNN time: %1.4f' % (stop - start))
    return pnet, rnet, onet



def align_data(image_paths, image_size, margin, pnet, rnet, onet):
    minsize = 80  # minimum size of face
    threshold = [0.6, 0.6, 0.7]  # three steps's threshold
    factor = 0.9  # scale factor

    # loadstart = clock()
    # print ('initializing net')
    # with tf.Graph().as_default():
    #     sess = tf.Session()
    #     with sess.as_default():
    #         pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    #
    # loadend = clock()
    #
    # print ('load mtcnn time cost: %1.4f' % (loadend - loadstart))

    img = misc.imread(image_paths)
    h = img.shape[0]
    w = img.shape[1]

    detectstart = clock()
    bounding_boxes, points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    detectend = clock()

    print ('detect face time cost: %1.4f' % (detectend - detectstart))

    num_of_face = bounding_boxes.shape[0]
    print("boundingboxes:%d" % num_of_face)
    #misc.imshow(img)

    img_size = np.asarray(img.shape)[0:2]

    bb = np.zeros(num_of_face * 4, dtype=np.int32)

    for i in range(num_of_face):
        det = bounding_boxes[i:i + 1, 0:4]
        det = np.squeeze(det)
        print("This is the %d face:" % (i + 1))
        print(det)
        bb[i * 4] = np.maximum(det[0] - margin / 2, 0)
        bb[i * 4 + 1] = np.maximum(det[1] - margin / 2, 0)
        bb[i * 4 + 2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[i * 4 + 3] = np.minimum(det[3] + margin / 2, img_size[0])

    tpimg = misc.imread(image_paths)

    fp = np.zeros(num_of_face * 10, dtype=np.int32)
    for i in range(num_of_face):
        fpoint = points[:, i]
        fpoint = np.squeeze(fpoint)

        for j in range(10):
            fp[i * 10 + j] = np.maximum(fpoint[j], 0)

        for j in range(5):
            tpimg = cv2.circle(tpimg, (fp[i * 10 + j], fp[i * 10 + j + 5]), 3, (0, 255, 0), 2, 8, 0)

    for i in range(num_of_face):
        tpimg = cv2.rectangle(tpimg, (bb[i * 4 + 0], bb[i * 4 + 1]), (bb[i * 4 + 2], bb[i * 4 + 3]), (255, 255, 0), 2, 8, 0)

    #misc.imshow(tpimg)

    img_list = [None] * num_of_face
    fpT = np.zeros(num_of_face * 10, dtype=np.int32)
    for i in range(num_of_face):
        anti = fp[i * 10 + 1] - fp[i * 10]
        neib = fp[i * 10 + 6] - fp[i * 10 + 5]
        radi = math.atan(neib / anti)
        angl = radi * 180 / math.pi

        if (angl != 0):
            direction = abs(angl) / angl
        else:
            direction = 1

        print(direction)

        rotatemat = cv2.getRotationMatrix2D((fp[i * 10], fp[i * 10 + 5]), angl, 1)
        ttimg = cv2.warpAffine(img, rotatemat, (w, h))

        for j in range(5):
            xT = fp[i * 10 + j] - fp[i * 10]
            yT = fp[i * 10 + 5 + j] - fp[i * 10 + 5]

            dx = float(xT) * math.cos(radi) - direction * float(yT) * math.sin(radi)
            dy = direction * float(xT) * math.sin(radi) + float(yT) * math.cos(radi)
            fpT[i * 10 + j] = dx + fp[i * 10]
            fpT[i * 10 + j + 5] = dy + fp[i * 10 + 5]

        dup = fpT[i * 10 + 1] - fpT[i * 10]
        ddown = fpT[i * 10 + 4] - fpT[i * 10 + 3]
        X = np.maximum(dup, ddown)

        dleft = fpT[i * 10 + 8] - fpT[i * 10 + 5]
        dright = fpT[i * 10 + 9] - fpT[i * 10 + 6]
        Y = np.maximum(dleft, dright)

        baseX = np.maximum(fpT[i * 10] - X * 0.5, 0)
        baseY = np.maximum(fpT[i * 10 + 5] - Y * 0.5, 0)
        cropped = ttimg[int(baseY):np.minimum(int(baseY + 2 * Y), h), int(baseX):np.minimum(int(baseX + 2 * X), w), :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
        #misc.imshow(prewhitened)

    if (num_of_face > 0):
        images = np.stack(img_list)
        return images, num_of_face, tpimg, bb

    else:
        return 0, num_of_face, tpimg, 0


def get_embedding(model_dir, aligned):
    images = aligned
    num_of_face = images.shape[0]

    with tf.Graph().as_default():
        with tf.Session() as sess:
            print ('Model directory: %s' % model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(model_dir)
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            start = clock()
            facenet.load_model(model_dir, meta_file, ckpt_file)
            stop = clock()
            print ("load model time: %1.4f" % (start-stop))

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            feed_dict = {images_placeholder: images, phase_train_placeholder: False}

            sta = clock()
            emb = sess.run(embeddings, feed_dict=feed_dict)
            end = clock()
            print ("emb time: %1.4f" % (sta - end))
            # print('Distance Matrix:')
            # print('    ', end='')
            # for i in range(num_of_face):
            #     print('    %ld    ' % i, end='')
            # print('')
            #
            # for i in range(num_of_face):
            #     print('%ld  ' % i, end='')
            #     for j in range(num_of_face):
            #         dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
            #         print('  %1.4f  ' % dist, end='')
            #     print('')

            return emb

def get_idimg(image_paths):
    files = os.listdir(image_paths)
    img_files = [s for s in files if s.endswith('.jpg')]
    num_of_id = len(img_files)
    print ("Length:%d" % num_of_id)
    img_list = [None] * num_of_id
    for i in range(num_of_id):
        timg = misc.imread(image_paths+'/'+img_files[i])
        #misc.imshow(img_list[i])
        prewhitened = facenet.prewhiten(timg)
        img_list[i] = prewhitened
        img_files[i] = img_files[i][0:(len(img_files[i])-4)]

    images = np.stack(img_list)
    return images, img_files, num_of_id

def get_idlist(testemb, idemb, face_in_test, ids):
    id_list = np.zeros(ids, dtype=np.int32)
    for i in range(face_in_test):
        for j in range(ids):
            dist = np.sqrt(np.sum(np.square(np.subtract(testemb[i, :], idemb[j, :]))))
            print ('dist from %d to %d: %1.4f' % (i, j, dist))
            if (dist < 1.2):
                id_list[j] = 1

    return id_list

def recognize(testemb, idemb, face_in_test, ids, detected_img, bb, id_name):
    for i in range(face_in_test):
        for j in range(ids):
            dist = np.sqrt(np.sum(np.square(np.subtract(testemb[i, :], idemb[j, :]))))
            print('dist from %d to %d: %1.4f' % (i, j, dist))
            if (dist < 1.2):
                font_size = float(bb[i*4+2] - bb[i*4])/200.0
                print ('font size: %1.4f' % font_size)
                bold = 2
                cv2.putText(detected_img, id_name[j], (bb[i*4], bb[i*4+1]-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 0), bold)

    return detected_img


def initFaceNet(model_dir):
    mysess = tf.Session()
    print ('init FaceNet')
    # print('Model directory: %s' % model_dir)
    meta_file, ckpt_file = facenet.get_model_filenames(model_dir)
    # print('Metagraph file: %s' % meta_file)
    # print('Checkpoint file: %s' % ckpt_file)

    start = clock()
    # facenet.load_model('/home/wst/Downloads/facenetmodel/20170216-091149', meta_file, ckpt_file)

    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(mysess, os.path.join(model_dir_exp, ckpt_file))
    stop = clock()
    print("load FaceNet time: %1.4f" % (stop - start))

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    return mysess, images_placeholder, embeddings, phase_train_placeholder



if __name__=="__main__":
    img, face_in_test, detected_img = align_data('./imgdata/ab12.jpg', 160, 32)
    idimg, id_name, ids = get_idimg('/home/wst/Downloads/fromgit/facenet/src/id')
    # misc.imshow(img[0])
    # misc.imshow(idimg[0])

    if (face_in_test > 0 and ids > 0):
        testemb = get_embedding('/home/wst/Downloads/facenetmodel/20170216-091149', img)
        idemb = get_embedding('/home/wst/Downloads/facenetmodel/20170216-091149', idimg)

        id_list = get_idlist(testemb, idemb, face_in_test, ids)

        reco_id = 0

        for i in range(ids):
            if (id_list[i] == 1):
                print ('%s is in the photo.' % id_name[i])
                reco_id += 1

        if (reco_id == 0):
            print ('nobody in database being recognized.')

    else:
        print ('no face detect.')

    misc.imshow(detected_img)