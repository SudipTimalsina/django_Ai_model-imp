
from django.contrib import admin
from django.urls import path, include
from home.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('home/', include('home.urls')),
]