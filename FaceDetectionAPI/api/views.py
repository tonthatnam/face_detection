import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from api.models.image import Image as Image_object
from api.serializers.image_serializer import ImageSerializer
import os
from django.conf import settings
from django.http import HttpResponse
from PIL import Image as PImage

from api.retinaface.test_fddb import detect_faces_retina, detect_faces_callback_retina

def upload_image(request, image_id):
    img = request.FILES['image']
    img_extension = os.path.splitext(img.name)[-1]
    return default_storage.save("image/" + image_id + img_extension, request.FILES['image'])

class Image(APIView):
    def post(self, request):
        image_serializer = ImageSerializer(data=request.data)
        if image_serializer.is_valid() and request.FILES.get("image", None):
            image_id = str(uuid.uuid4())
            name = upload_image(request, image_id)
            image = Image_object()
            image.image_id = image_id
            image.name = name
            image.save()
            detected_faces = detect_faces_retina(image_id)
            detect_faces_callback_retina(image_id)
            return Response({"status":detected_faces}, status=status.HTTP_202_ACCEPTED)
        else:
            if not request.FILES.get("image", None):
                return Response({'`image` is required'}, status=status.HTTP_400_BAD_REQUEST)
            return Response(image_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
