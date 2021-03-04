from rest_framework import serializers

from api.models.image import Image


class ImageSerializer(serializers.ModelSerializer):
    class Meta():
        model = Image
        fields = ('status', 'date_created')
