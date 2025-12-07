from django.shortcuts import render
import cv2
import numpy as np
import random
import math
from .forms import imageuploadform
from .models import UploadedImage
import requests
from io import BytesIO
import cloudinary
import cloudinary.uploader
import os
from django.conf import settings

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

def home(request):
    return render(request, 'home.html')

# Helper function to download image from Cloudinary URL
def download_image_from_url(url):
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download image")
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img

# Helper function to upload processed image to Cloudinary
def upload_to_cloudinary(img_array, suffix):
    
    _, buffer = cv2.imencode('.jpg', img_array)
    result = cloudinary.uploader.upload(
        BytesIO(buffer),
        folder="processed_images",
        resource_type="image"
    )
    return result['secure_url']

# convert to sketch edges
def convert_to_sketch_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    edges = cv2.Canny(blurred, 50, 150) 
    edges = cv2.dilate(edges, None, iterations=2)
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_3channel = 255 - edges_3channel  
    
    background = np.full_like(edges_3channel, (230, 230, 230), dtype=np.uint8)
    mask = np.all(edges_3channel == [0, 0, 0], axis=-1)
    background[mask] = [0, 0, 0]  

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    sketch = cv2.bitwise_and(background, img_gray)
    sketch = cv2.addWeighted(sketch, 0.7, background, 0.3, 0)
    
    return sketch

# convert to pen sketch
def convert_to_sketch_edges1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth_gray = cv2.bilateralFilter(gray, 9, 75, 75)
    inverted = 255 - smooth_gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch_base = cv2.divide(gray, 255 - blurred, scale=256)
    adaptiveedges = cv2.adaptiveThreshold(sketch_base, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    kernel = np.ones((1, 1), np.uint8)
    adaptive_edges = cv2.dilate(adaptiveedges, kernel, iterations=1)
    sketch = cv2.cvtColor(adaptiveedges, cv2.COLOR_GRAY2BGR)
    return sketch

# convert to pencil sketch
def convert_to_sketch_pencil(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    inverted_blur = 255 - blurred
    dodge = cv2.divide(gray, inverted_blur, scale=256)
    
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY_INV)[1]
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    edges = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=1)
    
    sketch = cv2.multiply(dodge, edges, scale=1/255.0)
    sketch = cv2.convertScaleAbs(sketch, alpha=1.7, beta=5)
    
    paper_texture = np.random.normal(loc=128, scale=4, size=sketch.shape).astype(np.uint8)
    paper_texture = cv2.GaussianBlur(paper_texture, (17, 17), 0)
    
    final = cv2.multiply(sketch, paper_texture, scale=1/255)
    final = np.clip(final, 0, 255).astype(np.uint8)
    return final

# convert to watercolor
def convert_to_watercolor(img):
    img = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
    water_color = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    
    paper = np.ones_like(water_color) * 240
    grain = np.random.normal(0, 2, paper.shape).astype(np.uint8)
    paper = cv2.subtract(paper, grain)
    paper = cv2.GaussianBlur(paper, (15, 15), 0)
    water_color = cv2.addWeighted(water_color, 0.85, paper, 0.15, 0)
    return water_color

# convert to oil painting
def convert_to_oil_painting(img):
    if img is None:
        return None
    stylized = cv2.edgePreservingFilter(img, flags=1, sigma_s=100, sigma_r=0.5)
    blur = cv2.bilateralFilter(stylized, 9, 75, 75)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    oil_paint = cv2.filter2D(blur, -1, sharpen_kernel)
    return oil_paint

# convert to charcoal
def convert_to_charcoal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY_INV)[1]
    
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    shadows = cv2.multiply(smoothed, edges, scale=1/255)
    
    grain = np.random.normal(loc=128, scale=10, size=shadows.shape).astype(np.uint8)
    grain = cv2.GaussianBlur(grain, (15, 15), 0)
    
    charcoal = cv2.addWeighted(shadows, 0.85, grain, 0.15, 10)
    charcoal = np.clip(charcoal, 0, 255).astype(np.uint8) 
    charcoal = cv2.convertScaleAbs(charcoal, alpha=1.4, beta=-30)
    return charcoal

# digital painting
def convert_to_digital_painting(img):
    img_bilateral = cv2.bilateralFilter(img, d=15, sigmaColor=100, sigmaSpace=100)
    blur = cv2.GaussianBlur(img_bilateral, (15, 15), 0)
    img_sharpened = cv2.addWeighted(img_bilateral, 1.5, blur, -0.5, 0)
    
    hsv = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = hsv[..., 1] * 1.5
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    img_saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    rows, cols = img_saturated.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/5)
    kernel_y = cv2.getGaussianKernel(rows, rows/5)
    kernel = kernel_y @ kernel_x.T
    vintage = np.zeros((rows, cols, 3), dtype=np.float32)
    normalized_kernel = kernel / np.max(kernel)
    
    for i in range(3):
        vintage[:, :, i] = normalized_kernel
    
    digital_painting = cv2.multiply(img_saturated.astype(np.float32), vintage.astype(np.float32))
    digital_painting = np.clip(digital_painting, 0, 255).astype(np.uint8)
    return digital_painting

# acrylic painting
def convert_to_acrylic_painting(img):
    img_bilateral = cv2.bilateralFilter(img, d=15, sigmaColor=100, sigmaSpace=100)
    img_blurred = cv2.GaussianBlur(img_bilateral, (11, 11), 0)
    img_sharpened = cv2.addWeighted(img_bilateral, 1.5, img_blurred, -0.5, 0)
    
    hsv = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = hsv[..., 1] * 1.5
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255).astype(np.uint8)
    img_saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    noise = np.random.normal(scale=5, size=img_saturated.shape).astype(np.uint8)
    img_textured = cv2.add(img_saturated, noise)
    acrylic_painting = np.clip(img_textured, 0, 255).astype(np.uint8)
    return acrylic_painting

# pen and ink drawing
def convert_to_pen_and_ink(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    
    _, ink_effect = cv2.threshold(sketch, 127, 255, cv2.THRESH_BINARY)
    noise = np.random.normal(0, 5, gray.shape).astype(np.uint8)
    ink_effect = cv2.add(ink_effect, noise)
    pen_and_ink = np.clip(ink_effect, 0, 255).astype(np.uint8)
    return pen_and_ink

# spray painting
def convert_to_spray_painting(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.add(hsv[..., 1], 40)
    img_saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    blurred = cv2.GaussianBlur(img_saturated, (11, 11), 3)
    noise = np.random.normal(0, 20, img.shape).astype(np.int16)
    noisy_img = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    splatter = np.zeros((h, w), dtype=np.uint8)
    for _ in range(300):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(1, 5)
        cv2.circle(splatter, (x, y), radius, 255, 1)
    
    splatter = cv2.GaussianBlur(splatter, (7, 7), 3)
    splatter_mask = cv2.cvtColor(splatter, cv2.COLOR_GRAY2BGR) / 255.0
    
    spray_paint = noisy_img.astype(np.float32) * (1 - splatter_mask) + splatter_mask * 255
    spray_paint = np.clip(spray_paint, 0, 255).astype(np.uint8)
    return spray_paint

# tattoo drawing
def convert_to_tattoo_drawing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(threshold_img, 100, 200)
    contrast_img = cv2.convertScaleAbs(edges, alpha=2.0, beta=50)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(contrast_img, kernel, iterations=2)
    blurred = cv2.GaussianBlur(dilated_edges, (3, 3), 0)
    
    tattoo_drawing = cv2.addWeighted(contrast_img, 0.7, blurred, 0.3, 0)
    tattoo_drawing = cv2.convertScaleAbs(tattoo_drawing, alpha=1.2, beta=0)
    return tattoo_drawing

# hatching drawing
def convert_to_hatching_drawing(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    hatching_image = np.ones((h, w, 3), dtype=np.uint8) * 255
    line_spacing = 10
    
    for y in range(0, h, line_spacing):
        for x in range(0, w, line_spacing):
            intensity = gray[y, x]
            
            if intensity < 150:
                line_density = int((150 - intensity) / 10)
                for i in range(line_density):
                    cv2.line(hatching_image, (x - 2, y + i * 2), (x + 30, y + i * 2), (0, 0, 0), 1)
            if intensity > 100:
                line_density = int((intensity - 100) / 10)
                for i in range(line_density):
                    angle = random.choice([45, 135])
                    length = random.randint(10, 20)
                    x_offset = int(math.cos(math.radians(angle)) * length)
                    y_offset = int(math.sin(math.radians(angle)) * length)
                    cv2.line(hatching_image, (x, y), (x + x_offset, y + y_offset), (0, 0, 0), 1)
    
    hatching_image[edges == 255] = [0, 0, 0]
    return hatching_image

# calligraphy pen drawing
def convert_to_calligraphy_drawing(img):
    brush_size = 5
    contrast = 1.5
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=contrast)
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    canvas = np.ones_like(img) * 255
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if smoothed[y, x]:
                cv2.line(canvas, (x - brush_size, y), (x + brush_size, y), (0, 0, 0), 1)
    return canvas

# 3d drawing
def convert_to_3d_drawing(img):
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    depth_map = cv2.GaussianBlur(gray, (21, 21), 0)
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.dilate(edges, None, iterations=1)
    edges_3channels = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_3channels = 255 - edges_3channels
    
    detail_enhanced = cv2.detailEnhance(original, sigma_s=100, sigma_r=0.15)
    stylized = cv2.stylization(detail_enhanced, sigma_s=60, sigma_r=0.07)
    
    result = cv2.addWeighted(stylized, 0.7, edges_3channels, 0.3, 0)
    h, w = result.shape[:2]
    light_effect = np.zeros((h, w, 3), dtype=np.uint8)
    light_effect = cv2.ellipse(light_effect, (int(w/2), int(h/3)), (int(w/3), int(h/3)), 
                               0, 0, 360, (200, 200, 200), -1)
    light_effect = cv2.GaussianBlur(light_effect, (51, 51), 0)
    result = cv2.addWeighted(result, 0.85, light_effect, 0.15, 0)
    
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def uploadImage(request):
    context = {
        'form': imageuploadform(),
        'original_url': None,
        'sketchurl1': None,
        'sketchurl2': None,
        'sketchurl3': None,
        'sketchurl4': None,
        'sketchurl5': None,
        'sketchurl6': None,
        'sketchurl7': None,
        'sketchurl8': None,
        'sketchurl9': None,
        'sketchurl10': None,
        'sketchurl11': None,
        'sketchurl12': None,
        'sketchurl13': None,
        'sketchurl14': None,
        'error_message': ''
    }

    if request.method == 'POST':
        form = imageuploadform(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                upload_image = form.save()
                original_url = upload_image.image.url
                context['original_url'] = original_url
                img = download_image_from_url(original_url)
                if img is None:
                    context['error_message'] = 'Failed to process the image.'
                    return render(request, 'upload.html', context)
                processed_images = {
                    'edges': convert_to_sketch_edges(img),
                    'edges1': convert_to_sketch_edges1(img),
                    'pencil': convert_to_sketch_pencil(img),
                    'watercolor': convert_to_watercolor(img),
                    'oil_painting': convert_to_oil_painting(img),
                    'charcoal': convert_to_charcoal(img),
                    'digital_painting': convert_to_digital_painting(img),
                    'acrylic_painting': convert_to_acrylic_painting(img),
                    'pen_and_ink': convert_to_pen_and_ink(img),
                    'spray_painting': convert_to_spray_painting(img),
                    'tattoo_drawing': convert_to_tattoo_drawing(img),
                    'hatching_drawing': convert_to_hatching_drawing(img),
                    'calligraphy': convert_to_calligraphy_drawing(img),
                    '3d_drawing': convert_to_3d_drawing(img)
                }
                context['sketchurl1'] = upload_to_cloudinary(processed_images['edges'], '_edges')
                context['sketchurl2'] = upload_to_cloudinary(processed_images['edges1'], '_edges1')
                context['sketchurl3'] = upload_to_cloudinary(processed_images['pencil'], '_pencil')
                context['sketchurl4'] = upload_to_cloudinary(processed_images['watercolor'], '_watercolor')
                context['sketchurl5'] = upload_to_cloudinary(processed_images['oil_painting'], '_oil_painting')
                context['sketchurl6'] = upload_to_cloudinary(processed_images['charcoal'], '_charcoal')
                context['sketchurl7'] = upload_to_cloudinary(processed_images['digital_painting'], '_digital_painting')
                context['sketchurl8'] = upload_to_cloudinary(processed_images['acrylic_painting'], '_acrylic_painting')
                context['sketchurl9'] = upload_to_cloudinary(processed_images['pen_and_ink'], '_pen_and_ink')
                context['sketchurl10'] = upload_to_cloudinary(processed_images['spray_painting'], '_spray_painting')
                context['sketchurl11'] = upload_to_cloudinary(processed_images['tattoo_drawing'], '_tattoo_drawing')
                context['sketchurl12'] = upload_to_cloudinary(processed_images['hatching_drawing'], '_hatching_drawing')
                context['sketchurl13'] = upload_to_cloudinary(processed_images['calligraphy'], '_calligraphy')
                context['sketchurl14'] = upload_to_cloudinary(processed_images['3d_drawing'], '_3d_drawing')
            except Exception as e:
                context['error_message'] = f'Error processing image: {str(e)}'
                return render(request, 'upload.html', context)
        else:
            context['form'] = form
    
    return render(request, 'upload.html', context)
    