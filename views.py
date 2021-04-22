from django.shortcuts import render
from .models import extractive

# Create your views here.
def home(request):
    return render(request,'search.html',{"Topic":"Enter Topic"})
def search(request):  
    
        Topic = request.GET["Topic"]
        if (len(Topic)==0):
            return render(request,'search.html',{"Topic":"Enter Topic"})
        else:
            p1 = extractive(Topic)
            return render(request,'search.html',{"Topic":Topic.upper(),"Summary":p1.sum})   

  
        