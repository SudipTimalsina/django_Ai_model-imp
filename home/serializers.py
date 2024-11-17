from rest_framework import serializers
from .models import birdlist
class birdlistSerializer(serializers.ModelSerializer):
    class Meta:
        model = birdlist
        fields = ['id','name','scientificName','image','birdUrl']