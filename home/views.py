from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def home(request):
    peoples = [
        {'name':'Sudip', 'age': 20},
        {'name':'Randeep', 'age': 22},
        {'name':'Prakash', 'age': 32},
        {'name':'Sandeep', 'age': 44},
        {'name':'Vicky kaushal', 'age': 34}
    ]

    for i in peoples:
        print(i)

    return render(request ,"home/index.html", context ={'peoples':peoples})
    

def success_page(request):
    return HttpResponse("Hey this is a success page")