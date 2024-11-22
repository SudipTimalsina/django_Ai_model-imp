from django.urls import path
from . import views

urlpatterns = [
    path('upload-audio/', views.upload_and_predict, name='upload_audio'),
]