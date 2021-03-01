Create a simple face detection REST API by using technologies below
 
 * Pytorch
 * Django
 * Docker
 * PostgreSQL

# Local Deploymnet
 
## Clone repository
 ```console
 git clone https://github.com/tonthatnam/face_detection.git
 cd face_detection
```
## Download pretrained model
 * Download pretraind model from Training section of [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
 * Organise the directory of pretrained model as follows:
```
    ./FaceDetectionAPI/api/models
```
## Docker build
```console
 docker-compose up
```
## Run rest
 To test REST API, send a post request to the end point http://localhost:8900/api/image/

## Articles and guides that cover face_detection

 * [Build a Production Ready Face Detection API](https://medium.com/devcnairobi/build-a-production-ready-face-detection-api-part-1-c56cbe9592bf) by Urandu Bildad Namawa
 * [Dockerizing Django with Postgres, Gunicorn, and Nginx](https://testdriven.io/blog/dockerizing-django-with-postgres-gunicorn-and-nginx/#gunicorn) by Michael Herman
 * [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)

