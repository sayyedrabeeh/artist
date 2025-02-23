from django import forms
from . models import UploadedImage

class imageuploadform(forms.ModelForm):
    class Meta:
        model=UploadedImage
        fields=['image']
    
