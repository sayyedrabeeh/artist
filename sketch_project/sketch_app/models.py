from django.db import models
from cloudinary.models import CloudinaryField

# Create your models here.

 

class UploadedImage(models.Model):
    image = CloudinaryField('image')
    uploaded_at = models.DateTimeField(auto_now_add=True)