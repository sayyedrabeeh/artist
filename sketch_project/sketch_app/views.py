from django.shortcuts import render
import cv2
import numpy as np
from .forms import imageuploadform
from .models import UploadedImage

def home(request):
    return render(request, 'home.html')

def convert_to_sketch_edges(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 120) 
    edges = cv2.dilate(edges, None, iterations=2)   
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_3channel = 255 - edges_3channel  
    
    background = np.full_like(edges_3channel, (200, 200, 200), dtype=np.uint8)
    
    
    mask = (edges_3channel == [0, 0, 0])
    for i in range(3):  
        background[:, :, i][mask[:, :, i]] = 0  

    sketch_path = image_path.replace('.jpg', '_edges.jpg')
    cv2.imwrite(sketch_path, background)
    return sketch_path

def convert_to_sketch_edges1(image_path):
    img = cv2.imread(image_path)
     
    height, width = img.shape[:2]
    max_dim = 1200
    if height > max_dim or width > max_dim:
        scale = max_dim / max(height, width)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
     
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    
    edges = cv2.Canny(gray, 10, 70)
    
     
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    
    edges_inverted = 255 - edges
    
    
    background = np.full_like(img, (245, 245, 245), dtype=np.uint8)
    
    
    edges_3channel = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR)
    
     
    alpha = 0.9
    sketch = cv2.addWeighted(background, 1 - alpha, edges_3channel, alpha, 0)
    
    sketch_path = image_path.replace('.jpg', '_edges1.jpg')
    cv2.imwrite(sketch_path, sketch)
    return sketch_path


def convert_to_sketch_pencil(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    
    sketch = cv2.convertScaleAbs(sketch, alpha=1.5, beta=5)

 
    
    sketch_path = image_path.replace('.jpg', '_pencil.jpg')
    cv2.imwrite(sketch_path, sketch)
    return sketch_path

def convert_to_sketch_pencil1(image_path):
    img = cv2.imread(image_path)
    
    height, width = img.shape[:2]
    max_dim = 1200
    if height > max_dim or width > max_dim:
        scale = max_dim / max(height, width)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
 
    smooth = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
    gray_smooth = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    
  
    inverted = 255 - gray_smooth
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
     
    sketch_div = cv2.divide(gray_smooth, 255 - blurred, scale=256)
    
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    
    thresh_inv = 255 - thresh
    
 
    thresh_blur = cv2.GaussianBlur(thresh_inv, (3, 3), 0)
    
    
    paper = np.ones_like(gray) * 235
    paper = paper.astype(np.uint8)
     
    grain = np.random.normal(0, 3, paper.shape).astype(np.uint8)
    paper = cv2.subtract(paper, grain)
 
    sketch = cv2.addWeighted(sketch_div, 0.6, thresh_blur, 0.4, 0)
 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    sketch = clahe.apply(sketch)
 
    kernel = np.ones((3, 3), np.float32)/25
    smudge = cv2.filter2D(sketch, -1, kernel)
    sketch = cv2.addWeighted(sketch, 0.7, smudge, 0.3, 0)
 
    rows, cols = sketch.shape
    kernel_x = cv2.getGaussianKernel(cols, cols/5)
    kernel_y = cv2.getGaussianKernel(rows, rows/5)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    sketch = sketch * (mask * 0.3 + 0.7)
    
  
    sketch = cv2.multiply(sketch.astype(np.float32)/255, paper.astype(np.float32)/255) * 255
    sketch = sketch.astype(np.uint8)
    
    sketch_path = image_path.replace('.jpg', '_pencil1.jpg')
    cv2.imwrite(sketch_path, sketch)
    return sketch_path

def convert_to_colored_pencil(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)

    
    color_pencil = cv2.applyColorMap(img, cv2.COLORMAP_JET)  
    blended = cv2.bitwise_and(color_pencil, color_pencil, mask=sketch)   
 
    output_path = image_path.replace('.jpg', '_pencil_colored.jpg')
    cv2.imwrite(output_path, blended)
    return output_path

def convert_to_colored_pencil1(image_path):
    img = cv2.imread(image_path)
    
 
    height, width = img.shape[:2]
    
     
    max_dim = 1200
    if height > max_dim or width > max_dim:
        scale = max_dim / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
        
        height, width = new_height, new_width
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
   
    smooth = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
    gray_smooth = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    
    
    inverted = 255 - gray_smooth
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    pencil_sketch = cv2.divide(gray_smooth, 255 - blurred, scale=256)
    
 
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    thresh_inv = 255 - thresh
    thresh_blur = cv2.GaussianBlur(thresh_inv, (3, 3), 0)
 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    hsv[:,:,1] = hsv[:,:,1] * 0.45
   
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)
    colored_base = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
     
    colored_base = cv2.detailEnhance(colored_base, sigma_s=10, sigma_r=0.15)
    
   
    paper = np.ones((height, width, 3), np.uint8) * np.array([235, 235, 230], dtype=np.uint8)
     
    grain = np.random.normal(0, 3, paper.shape).astype(np.uint8)
    paper = cv2.subtract(paper, grain)
    
    
    pencil_3channel = cv2.cvtColor(pencil_sketch, cv2.COLOR_GRAY2BGR)
    
    thresh_3channel = cv2.cvtColor(thresh_blur, cv2.COLOR_GRAY2BGR)
    
 
    colored_pencil = cv2.multiply(colored_base.astype(np.float32)/255, 
                                  pencil_3channel.astype(np.float32)/255) * 255
    colored_pencil = colored_pencil.astype(np.uint8)
    
 
    alpha = 0.65   
    beta = 0.4     
    gamma = 0.0    
    result = cv2.addWeighted(colored_pencil, alpha, thresh_3channel, beta, gamma)
    
   
    stroke_texture = np.zeros_like(result)
    for i in range(3):
        noise = np.random.normal(0, 5, (height, width)).astype(np.uint8)
        noise_blur = cv2.GaussianBlur(noise, (5, 5), 0)
        stroke_texture[:,:,i] = noise_blur
    
    
    result = cv2.addWeighted(result, 0.9, stroke_texture, 0.1, 0)
    
 
    result = cv2.multiply(result.astype(np.float32)/255, paper.astype(np.float32)/255) * 255
    result = result.astype(np.uint8)
    
   
    rows, cols = result.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/5)
    kernel_y = cv2.getGaussianKernel(rows, rows/5)
    kernel = kernel_y * kernel_x.T
    
    mask = np.zeros((rows, cols, 3), dtype=np.float32)
    normalized_kernel = 255 * kernel / np.linalg.norm(kernel)
    for i in range(3):
        mask[:,:,i] = normalized_kernel
    
    
    result = result * (mask * 0.3 + 0.7)
    result = result.astype(np.uint8)
    
    output_path = image_path.replace('.jpg', '_pencil_colored1.jpg')
    cv2.imwrite(output_path, result)
    return output_path


 
def uploadImage(request):
    sketchurl1 = None   
    sketchurl2 = None  
    sketchurl3 = None  
    sketchurl4 = None  
    sketchurl5 = None  
    sketchurl6 = None  
    original_url=None
    
    if request.method == 'POST':
        form = imageuploadform(request.POST, request.FILES)
        image_file = request.FILES.get('image_file') 
        if form.is_valid():
            upload_image = form.save()
            original_url = upload_image.image.url 
            
           
            sketch_path1 = convert_to_sketch_edges(upload_image.image.path)  
            sketch_path2 = convert_to_sketch_pencil(upload_image.image.path)  
            sketch_path3 = convert_to_sketch_pencil1(upload_image.image.path)  
            sketch_path4 = convert_to_sketch_edges1(upload_image.image.path)  
            sketch_path5 = convert_to_colored_pencil(upload_image.image.path)  
            sketch_path6 = convert_to_colored_pencil1(upload_image.image.path)  
            
            
            sketchurl1 = upload_image.image.url.replace('.jpg', '_edges.jpg')
            sketchurl2 = upload_image.image.url.replace('.jpg', '_edges1.jpg')
            sketchurl3 = upload_image.image.url.replace('.jpg', '_pencil.jpg')
            sketchurl4 = upload_image.image.url.replace('.jpg', '_pencil1.jpg')
            sketchurl5 = upload_image.image.url.replace('.jpg', '_pencil_colored.jpg')
            sketchurl6 = upload_image.image.url.replace('.jpg', '_pencil_colored1.jpg')
    
    else:
        form = imageuploadform()
    
    return render(request, 'upload.html', {
        'form': form,
        'original_url': original_url,
        'sketchurl1': sketchurl1,
        'sketchurl2': sketchurl2,
        'sketchurl3': sketchurl3,
        'sketchurl4': sketchurl4,
        'sketchurl5': sketchurl5,
        'sketchurl6': sketchurl6,
    })


