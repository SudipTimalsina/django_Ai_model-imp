from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_and_predict, name='upload_audio'),
]
# urlpatterns = [
#     path('predict/', views.predict_audio, name='predict_audio'),
#     # path('record', views.record_audio, name='record_audio'),
# ]