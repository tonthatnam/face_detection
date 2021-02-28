from __future__ import print_function
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from api.models.image import Image as Image_object
from api.serializers.image_serializer import ImageSerializer
#from api.tasks.image import detect_faces, detect_faces_callback
from api.tasks.image import detect_faces
import os
from django.conf import settings
import magic
from django.http import HttpResponse
from PIL import Image as PImage

####
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from api.retinaface.data.config import cfg_mnet, cfg_re50
from api.retinaface.layers.functions.prior_box import PriorBox
from api.retinaface.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from api.retinaface.models.retinaface import RetinaFace
from api.retinaface.utils.box_utils import decode, decode_landm
from api.retinaface.utils.timer import Timer
from api.retinaface.models.retinaface import RetinaFace
from api.retinaface.test_fddb import check_keys
from api.retinaface.test_fddb import remove_prefix
from api.retinaface.test_fddb import load_model
####

def upload_image(request, image_id):

    img = request.FILES['image']
    img_extension = os.path.splitext(img.name)[-1]
    return default_storage.save("image/" + image_id + img_extension, request.FILES['image'])


class Image(APIView):

    def post(self, request):
        ########################################################################
        trained_model = '/home/nam/デスクトップ/face_detection/face_detect_retinaface/Pytorch_Retinaface_test/weights/Resnet50_Final.pth'
        network ='resnet50'
        save_folder ='eval/'
        cpu = False
        dataset ='/home/nam/デスクトップ/face_detection/face_detect_retinaface/Pytorch_Retinaface_test/data/FDDB'
        dataset_name = 'FDDB'
        confidence_threshold = 0.8
        top_k = 5000
        nms_threshold = 0.3
        keep_top_k = 750

        image_serializer = ImageSerializer(data=request.data)
        if image_serializer.is_valid() and request.FILES.get("image", None):
            image_id = str(uuid.uuid4())
            name = upload_image(request, image_id)
            print(name)
            image = Image_object()
            image.image_id = image_id
            image.name = name
            image.save()
            ####################################################################
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
            cudnn.benchmark = True
            device = torch.device("cpu" if cpu else "cuda")
            net = net.to(device)

            # testing scale
            resize = 1

            _t = {'forward_pass': Timer(), 'misc': Timer()}

            deteted_faces2 = []
            # testing begin
            #for i, img_name in enumerate(test_dataset):
            if True:
                image_path = "/home/nam/デスクトップ/machine_service/my_ml_service_retinaFace/FaceDetectionAPI/face_detect_api/face_detect_api/" + name
                img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = np.float32(img_raw)
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

                print("-----------------------test_here_testBegin-----------------------")
                # save dets
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
                        deteted_faces2.append({"face_id": "face_id_1", "confidence": score, "bouding_box": [int(xmin), int(ymin), int(w), int(h)]})
                #print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

            return Response({"success":deteted_faces2}, status=status.HTTP_202_ACCEPTED)
        else:
            if not request.FILES.get("image", None):
                return Response({'`image` is required'}, status=status.HTTP_400_BAD_REQUEST)
            return Response(image_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
