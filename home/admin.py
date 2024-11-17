from django.contrib import admin
from .models import birdlist

# Register your models here.

@admin.register(birdlist)
class birdlistAdmin(admin.ModelAdmin):
    list_display = ['id','name','scientificName','image','birdUrl']
