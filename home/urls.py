from django.urls import path
from home import views

urlpatterns = [
    path('birdlist/',views.birdlistList.as_view()),
]