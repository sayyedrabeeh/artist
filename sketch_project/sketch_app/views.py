from django.shortcuts import render
import cv2
import numpy as np
from .forms import imageuploadform
from .models import UploadedImage

def home(request):
    return render(request, 'home.html')

# 1️⃣ Edge-based Sketch (Black outlines on gray background)
def convert_to_sketch_edges(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 120)  # Detect edges
    edges = cv2.dilate(edges, None, iterations=2)  # Thicken edges
    
    # Convert to 3 channels
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_3channel = 255 - edges_3channel  # Black edges on white
    
    # Create a gray background
    background = np.full_like(edges_3channel, (200, 200, 200), dtype=np.uint8)
    
    # Apply the black edges onto the gray background
    mask = (edges_3channel == [0, 0, 0])
    for i in range(3):  
        background[:, :, i][mask[:, :, i]] = 0  

    sketch_path = image_path.replace('.jpg', '_edges.jpg')
    cv2.imwrite(sketch_path, background)
    return sketch_path

# 2️⃣ Pencil Sketch (Soft shading effect)
def convert_to_sketch_pencil(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    
    sketch = cv2.convertScaleAbs(sketch, alpha=1.2, beta=10)
    
    sketch_path = image_path.replace('.jpg', '_pencil.jpg')
    cv2.imwrite(sketch_path, sketch)
    return sketch_path

# Image Upload & Processing
def uploadImage(request):
    sketchurl1 = None  # Edge-based sketch
    sketchurl2 = None  # Pencil sketch
    original_url=None
    
    if request.method == 'POST':
        form = imageuploadform(request.POST, request.FILES)
        image_file = request.FILES.get('image_file') 
        if form.is_valid():
            upload_image = form.save()
            original_url = upload_image.image.url 
            
            # Generate two versions of the sketch
            sketch_path1 = convert_to_sketch_edges(upload_image.image.path)  # Edge-based
            sketch_path2 = convert_to_sketch_pencil(upload_image.image.path)  # Pencil
            
            # Generate URLs
            sketchurl1 = upload_image.image.url.replace('.jpg', '_edges.jpg')
            sketchurl2 = upload_image.image.url.replace('.jpg', '_pencil.jpg')
    
    else:
        form = imageuploadform()
    
    return render(request, 'upload.html', {
        'form': form,
        'original_url': original_url,
        'sketchurl1': sketchurl1,
        'sketchurl2': sketchurl2
    })
