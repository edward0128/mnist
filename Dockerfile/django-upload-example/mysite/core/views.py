from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy

from .forms import BookForm
from .models import Book
import os
import re
import subprocess
class Home(TemplateView):
    template_name = 'home.html'


def upload(request):
    model_name = os.environ['MODEL_NAME']
    #model_name="bad_model"
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        img_path = fs.url(name)
        batcmd="python /mnt/"
        batcmd+=model_name
        batcmd+="/number.py "
        batcmd+="/tf/django-upload-example"
        batcmd+=img_path
        result = subprocess.check_output(batcmd, shell=True)
        result_test=str(result)
        result_test=result_test.replace('b', '')
        result_test=result_test.replace('\\n', '')
        result_test=result_test.replace("'","")
        context['url'] = fs.url(name)
        context['result']=str(result_test)
    return render(request, 'upload.html', context)

def book_list(request):
    books = Book.objects.all()
    return render(request, 'book_list.html', {
        'books': books
    })


def upload_book(request):
    if request.method == 'POST':
        form = BookForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('book_list')
    else:
        form = BookForm()
    return render(request, 'upload_book.html', {
        'form': form
    })


def delete_book(request, pk):
    if request.method == 'POST':
        book = Book.objects.get(pk=pk)
        book.delete()
    return redirect('book_list')


class BookListView(ListView):
    model = Book
    template_name = 'class_book_list.html'
    context_object_name = 'books'


class UploadBookView(CreateView):
    model = Book
    form_class = BookForm
    success_url = reverse_lazy('class_book_list')
    template_name = 'upload_book.html'
