![apm](https://img.shields.io/apm/l/vim-mode.svg) 

Create a simple face detection REST API by using technologies below

 * Pytorch
 * Django
 * Docker
 * PostgreSQL

Use pretrained model of [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) to detect faces in picture
## Local Deploymnet
#### Contents
- [Installation](#Installation)
- [Model](#model)
- [Build](#build)
- [Run](#run)
- [Result](#result)
- [References](#references)

## Installation
#### Clone repository
 ```console
 git clone https://github.com/tonthatnam/face_detection.git
 cd face_detection
```
## Model
#### Download pretrained model
 * Download pre-traind model from Training section of [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
 * Organise the directory of pretrained model as follows:
```
    ./FaceDetectionAPI/api/retinaface/weights/
        mobilenet0.25_Final.pth
        mobilenetV1X0.25_pretrain.tar
        Resnet50_Final.pth
```
## Build
#### Docker build
```console
 docker-compose up
```
## Run
#### Run test on local
 To test REST API, send a post request to the end point http://localhost:8900/api/image/

## Result
#### Find faces in pictures
##### Input image

##### Output image

#### Get response
```
{
    "status": [
        {
            "face_id": "3cb138bc-c268-4c46-95b3-9bc58eea3d91",
            "confidence": 0.9999927282333374,
            "bouding_box": [
                593,
                376,
                565,
                714
            ]
        }
    ]
}
```
## References
#### Articles and guides that cover face_detection

 * [Build a Production Ready Face Detection API](https://medium.com/devcnairobi/build-a-production-ready-face-detection-api-part-1-c56cbe9592bf) by Urandu Bildad Namawa
 * [Dockerizing Django with Postgres, Gunicorn, and Nginx](https://testdriven.io/blog/dockerizing-django-with-postgres-gunicorn-and-nginx/#gunicorn) by Michael Herman
 * [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) by biubug6
 * [face_recognition](https://github.com/ageitgey/face_recognition) by ageitgey
