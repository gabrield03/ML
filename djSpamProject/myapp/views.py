from django.shortcuts import render, HttpResponse
from .models import Email

# Create your views here.
def home(request):
    #return HttpResponse("hello world!")
    return render(request, "index.html")

# def todos(request):
#     items = Email.objects.all()
#     return render(request, "todos.html", {"todos": items})