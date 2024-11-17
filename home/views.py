from django.shortcuts import render
from .models import birdlist
from .serializers import birdlistSerializer
from rest_framework.generics import ListAPIView

# Create your views here.
class birdlistList(ListAPIView):
    queryset = birdlist.objects.all()
    serializer_class = birdlistSerializer
