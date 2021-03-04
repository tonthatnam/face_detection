from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from api.models.image import Image as Image_object
from PIL import Image as PImage
from api.retinaface.data import cfg_mnet, cfg_re50
from api.retinaface.layers.functions.prior_box import PriorBox
from api.retinaface.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
import json
import io

from api.retinaface.models.retinaface import RetinaFace
from api.retinaface.utils.box_utils import decode, decode_landm
from api.retinaface.utils.timer import Timer
import uuid

import requests
from django.core.files.storage import default_storage
from numpy import asarray
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.conf import settings

from api.models.face import Face

def detect_faces_retina(image_id):
    trained_model = '/workspace/api/retinaface/weights/Resnet50_Final.pth'
    network ='resnet50'
    save_folder ='eval/'
    #cpu = False
    cpu = True
    dataset_name = 'FDDB'
    confidence_threshold = 0.8
    top_k = 5000
    nms_threshold = 0.3
    keep_top_k = 750
    # testing scale
    resize = 1
    if True:
        image_object = Image_object.objects.get(image_id=image_id)
        filename = image_object.name
        image = PImage.open(default_storage.open(filename))
        image = image.convert('RGB')
        img = asarray(image)
        img = np.float32(img)

        torch.set_grad_enabled(False)
        cfg = None
        if network == "mobile0.25":
            cfg = cfg_mnet
        elif network == "resnet50":
            cfg = cfg_re50
        # net and model
        net = RetinaFace(cfg=cfg, phase = 'test')
        net = load_model(net, trained_model, cpu)
        net.eval()
        print('Finished loading model!')
        print(net)
        cudnn.benchmark = False
        device = torch.device("cpu" if cpu else "cuda")
        net = net.to(device)
        _t = {'forward_pass': Timer(), 'misc': Timer()}
        detected_faces2 = []
        if True:
            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            _t['forward_pass'].tic()
            loc, conf, landms = net(img)  # forward pass
            _t['forward_pass'].toc()
            _t['misc'].tic()
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            # order = scores.argsort()[::-1][:args.top_k]
            order = scores.argsort()[::-1]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            dets = dets[keep, :]
            landms = landms[keep]
            dets = np.concatenate((dets, landms), axis=1)
            _t['misc'].toc()
            if dataset_name == "FDDB":
                #fw.write('{:s}\n'.format(img_name))
                #fw.write('{:.1f}\n'.format(dets.shape[0]))
                for k in range(dets.shape[0]):
                    xmin = dets[k, 0]
                    ymin = dets[k, 1]
                    xmax = dets[k, 2]
                    ymax = dets[k, 3]
                    score = dets[k, 4]
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1
                    # fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(xmin, ymin, w, h, score))
                    #fw.write('{:d} {:d} {:d} {:d} {:.10f}\n'.format(int(xmin), int(ymin), int(w), int(h), score))
                    #save face data to DB
                    face_object = Face()
                    face_id = str(uuid.uuid4())
                    face_object.face_id = face_id
                    face_object.image_id = image_id
                    face_object.confidence = score
                    face_object.box = [int(xmin), int(ymin), int(w), int(h)]
                    face_object.save()
                    detected_faces2.append({"face_id": face_id, "confidence": score, "bouding_box": [int(xmin), int(ymin), int(w), int(h)]})
            #print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))
    return detected_faces2

def detect_faces_callback_retina(image_id):
    image_object = Image_object.objects.get(image_id=image_id)

    filename = image_object.name
    output_filename = "detected_faces/" + image_object.name
    faces_on_image = Face.objects.filter(image_id=image_id)
    image = PImage.open(default_storage.open(filename))
    image = np.array(image)
    image = image.copy()
    faces_dict = list()
    for face in faces_on_image:
        faces_dict.append({
            "confidence":face.confidence,
            "box":face.box,
        })
        box = json.loads(face.box)

        x1, y1, width, height = box
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(image,
                    "P: " + "{0:.4f}".format(float(face.confidence)),
                    (x1, (y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    pil_im = PImage.fromarray(image)
    silver_bullet = io.BytesIO()
    pil_im.save(silver_bullet, format="JPEG")

    image_file = InMemoryUploadedFile(silver_bullet, None, output_filename, 'image/jpeg',
                                      len(silver_bullet.getvalue()), None)

    default_storage.save(output_filename, image_file)

    return image_id



def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
