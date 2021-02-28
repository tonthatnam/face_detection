import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from api.models.image import Image as Image_object
from api.serializers.image_serializer import ImageSerializer
from api.tasks.image import detect_faces, detect_faces_callback
import os
from django.conf import settings
import magic
from django.http import HttpResponse
from PIL import Image as PImage


def upload_image(request, image_id):

    img = request.FILES['image']
    img_extension = os.path.splitext(img.name)[-1]
    return default_storage.save("image/" + image_id + img_extension, request.FILES['image'])


class Image(APIView):

    def post(self, request):
        image_serializer = ImageSerializer(data=request.data)
        if image_serializer.is_valid() and request.FILES.get("image", None):
            image_id = str(uuid.uuid4())
            #request_id = request.data.get("request_id")
            #callback_url = request.data.get("callback_url")
            name = upload_image(request, image_id)
            image = Image_object()
            image.image_id = image_id
            #image.request_id = request_id
            #image.callback_url = callback_url
            image.name = name
            image.save()


            detect_faces(image_id)
            detect_faces_callback(image_id)

            return Response({"status":"ok"}, status=status.HTTP_202_ACCEPTED)
        else:
            if not request.FILES.get("image", None):
                return Response({'`image` is required'}, status=status.HTTP_400_BAD_REQUEST)
            return Response(image_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
