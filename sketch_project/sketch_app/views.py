from django.shortcuts import render
import cv2
import numpy as np
from .forms import imageuploadform
from .models import UploadedImage

def home(request):
    return render(request, 'home.html')

# convert to sketch edges

def convert_to_sketch_edges(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred =cv2.GaussianBlur(gray,(21,21),0)

    edges = cv2.Canny(blurred, 50, 150) 
    edges = cv2.dilate(edges, None, iterations=2)

    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_3channel = 255 - edges_3channel  
    
    background = np.full_like(edges_3channel, (230, 230, 230), dtype=np.uint8)
    
    
    mask = np.all(edges_3channel == [0, 0, 0], axis=-1)
    background[mask] = [0,0,0]  

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)

    sketch = cv2.bitwise_and(background,img_gray)

    sketch = cv2.addWeighted(sketch,0.7,background,0.3,0)

    sketch_path = image_path.replace('.jpg', '_edges.jpg')
    cv2.imwrite(sketch_path, sketch)

    return sketch_path

# convert to pen sketch

def convert_to_sketch_edges1(image_path):
    img = cv2.imread(image_path)
     
    height, width = img.shape[:2]
    max_dim = 1200
    if height > max_dim or width > max_dim:
        scale = max_dim / max(height, width)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
     
    smooth_gray = cv2.bilateralFilter(gray, 9, 75, 75)

    inverted = 255-smooth_gray
    
    blurred = cv2.GaussianBlur(inverted,(21,21),0)

    sketch_base = cv2.divide(gray,255-blurred,scale=256)
    
    adaptiveedges = cv2.adaptiveThreshold(sketch_base,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,5)


     
    kernel = np.ones((1, 1), np.uint8)
    adaptive_edges = cv2.dilate(adaptiveedges, kernel, iterations=1)
    
    sketch = cv2.cvtColor(adaptiveedges,cv2.COLOR_GRAY2BGR)
    
    sketch_path = image_path.replace('.jpg', '_edges1.jpg')
    cv2.imwrite(sketch_path, sketch)
    return sketch_path


#  convert to pencil sketch 


def convert_to_sketch_pencil(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height,width = gray.shape
    max_dim =1200
    if height > max_dim or width > max_dim:
        scale = max_dim/max(height,width)
        img = cv2.resize(img,(int(width*scale),int(height*scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    inverted_blur = 255 - blurred
    dodge  = cv2.divide(gray, inverted_blur, scale=256)

    edges = cv2.Laplacian(gray,cv2.CV_8U,ksize=5)
    edges = cv2.threshold(edges,30,255,cv2.THRESH_BINARY_INV)[1]
    edges = cv2.GaussianBlur(edges,(3,3),0)
    edges = cv2.dilate(edges,np.ones((1,1),np.uint8),iterations = 1 )


    sketch = cv2.multiply(dodge,edges,scale=1/255.0)
    sketch = cv2.convertScaleAbs(sketch,alpha=1.7,beta=5)

    papper_tuxture = np.random.normal(loc=128,scale=4,size=sketch.shape).astype(np.uint8)
    papper_tuxture = cv2.GaussianBlur(papper_tuxture,(17,17),0)

    final = cv2.multiply(sketch,papper_tuxture,scale=1/255)
    final= np.clip(final,0,255).astype(np.uint8)
     

    sketch_path = image_path.replace('.jpg', '_pencil.jpg')
    cv2.imwrite(sketch_path, final)
    return sketch_path

#  convert to watercolor

def convert_to_watercolor(image_path):
    img = cv2.imread(image_path)
    
    height, width = img.shape[:2]
    max_dim = 1200
    if height > max_dim or width > max_dim:
        scale = max_dim / max(height, width)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))

    img = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
    water_color = cv2.stylization(img,sigma_s=60,sigma_r=0.6)

    paper = np.ones_like(water_color)*240
    grain = np.random.normal(0,2,paper.shape).astype(np.uint8)
    paper =cv2.subtract(paper,grain)

    paper = cv2.GaussianBlur(paper,(15,15),0)
    water_color = cv2.addWeighted(water_color,0.85,paper,0.15,0)

    
    sketch_path = image_path.replace('.jpg', '_watercolor.jpg')
    cv2.imwrite(sketch_path, water_color)
    return sketch_path

# convert to oil painting

def convert_to_oil_painting(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    h,w = img.shape[:2]
    if max(h,w) > 1200:
      scale = 1200/max(h,w)
      img = cv2.resize(img,(int(h*scale),int(w*scale)))

    stylezied = cv2.edgePreservingFilter(img,flags=1,sigma_s=100,sigma_r=0.5)
    blur = cv2.bilateralFilter(stylezied,9,75,75)
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5,-1],
                               [0, -1, 0]])
    oil_paint = cv2.filter2D(blur,-1,sharpen_kernel)

    output_path = image_path.replace('.jpg', '_oil_paintig.jpg')
    cv2.imwrite(output_path, oil_paint)
    return output_path

 

# convert to charcoal

def convert_to_charcoal(image_path):
    img = cv2.imread(image_path)
    h,w = img.shape[:2]
    if max(h,w)> 1200:
        scale = 1200/max(h,w)
        img = cv2.resize(img,(int(h*scale),int(w*scale)))
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = cv2.Laplacian(gray,cv2.CV_8U,ksize=5)
    edges = cv2.GaussianBlur(edges,(3,3),0)
    edges = cv2.threshold(edges,30,255,cv2.THRESH_BINARY_INV)[1]

    smoothed = cv2.bilateralFilter(gray,9,75,75)
    shadows = cv2.multiply(smoothed,edges,scale=1/255)

    grain = np.random.normal(loc=128,scale=10,size= shadows.shape).astype(np.uint8)
    grain = cv2.GaussianBlur(grain,(15,15),0)


    charcoal = cv2.addWeighted(shadows,0.85,grain,0.15,10)
    charcoal = np.clip(charcoal,0,255).astype(np.uint8) 

    charcoal = cv2.convertScaleAbs(charcoal,alpha=1.4,beta=-30)

  
    output_path = image_path.replace('.jpg', 'charcoal.jpg')
    cv2.imwrite(output_path, charcoal)
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
            sketch_path3 = convert_to_watercolor(upload_image.image.path)  
            sketch_path4 = convert_to_sketch_edges1(upload_image.image.path)  
            sketch_path5 = convert_to_oil_painting(upload_image.image.path)  
            sketch_path6 = convert_to_charcoal(upload_image.image.path)  
            
            
            sketchurl1 = upload_image.image.url.replace('.jpg', '_edges.jpg')
            sketchurl2 = upload_image.image.url.replace('.jpg', '_edges1.jpg')
            sketchurl3 = upload_image.image.url.replace('.jpg', '_pencil.jpg')
            sketchurl4 = upload_image.image.url.replace('.jpg', '_watercolor.jpg')
            sketchurl5 = upload_image.image.url.replace('.jpg', '_oil_paintig.jpg')
            sketchurl6 = upload_image.image.url.replace('.jpg', 'charcoal.jpg')
    
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


