from django.shortcuts import render
import cv2
import numpy as np
import random
import math
from .forms import imageuploadform
from .models import UploadedImage

def home(request):
    return render(request, 'home.html')

# convert to sketch edges

def convert_to_sketch_edges(image_path):
    if not image_path.lower().endswith('.jpg'):
        return None 
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
    if not image_path.lower().endswith('.jpg'):
        return None 
    img = cv2.imread(image_path)
     
    height, width = img.shape[:2]
    max_dim = 1200
    # if height > max_dim or width > max_dim:
    #     scale = max_dim / max(height, width)
    #     img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
 
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
    if not image_path.lower().endswith('.jpg'):
        return None 
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height,width = gray.shape
    max_dim =1200
    # if height > max_dim or width > max_dim:
    #     scale = max_dim/max(height,width)
    #     img = cv2.resize(img,(int(width*scale),int(height*scale)))
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
    if not image_path.lower().endswith('.jpg'):
        return None 
    img = cv2.imread(image_path)
    
    height, width = img.shape[:2]
    max_dim = 1200
    # if height > max_dim or width > max_dim:
    #     scale = max_dim / max(height, width)
    #     img = cv2.resize(img, (int(width * scale), int(height * scale)))

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
    if not image_path.lower().endswith('.jpg'):
        return None 
    img = cv2.imread(image_path)
    if img is None: return None
    h,w = img.shape[:2]
    # if max(h,w) > 1200:
    #   scale = 1200/max(h,w)
    #   img = cv2.resize(img,(int(h*scale),int(w*scale)))

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
    if not image_path.lower().endswith('.jpg'):
        return None 
    img = cv2.imread(image_path)
    h,w = img.shape[:2]
    # if max(h,w)> 1200:
    #     scale = 1200/max(h,w)
    #     img = cv2.resize(img,(int(h*scale),int(w*scale)))
    
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


# digital pinting 

def convert_to_digital_painting(image_path):
    if not image_path.lower().endswith('.jpg'):
        return None 
    img = cv2.imread(image_path)
    h,w = img.shape[:2]
    # if max(h,w) > 1200:
    #     scale =  1200/max(h,w)
    #     img= cv2.resize(img,(int(w*scale),int(h*scale)))
    h, w = img.shape[:2]
    
    img_biltaral = cv2.bilateralFilter(img,d=15,sigmaColor=100,sigmaSpace=100)

    blur = cv2.GaussianBlur(img_biltaral,(15,15),0)

    img_sharpened = cv2.addWeighted(img_biltaral,1.5,blur,-0.5,0)

    hsv = cv2.cvtColor(img_sharpened,cv2.COLOR_BGR2HSV)
    hsv[...,1]=hsv[...,1]*1.5
    hsv[...,1]= np.clip(hsv[...,1],0,255)
    img_staturated = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    rows,cols =img_staturated.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols,cols/5)
    kernel_y = cv2.getGaussianKernel(rows,rows/5)
    kernel = kernel_y @ kernel_x.T
    vintage = np.zeros((rows,cols,3) ,dtype=np.float32)
    normalaized_kernel = kernel/np.max(kernel)

    for i in range(3):
        vintage[:,:,i]=normalaized_kernel

    digital_painting = cv2.multiply(img_staturated.astype(np.float32),vintage.astype(np.float32))
    digital_painting =  np.clip(digital_painting,0,255).astype(np.uint8)


    output_path = image_path.replace('.jpg','digital_painting.jpg')
    cv2.imwrite(output_path,digital_painting)
    return output_path
 
# acrylic_painting

def convert_to_acrylic_painting(image_path):
    if not image_path.lower().endswith('.jpg'):
        return None 
    img = cv2.imread(image_path)
    h,w= img.shape[:2]
    # if max(h,w) > 1200:
    #     scale = 1200/max(h,w)
    #     img = cv2.resize(img,(int(h*scale),int(w*scale)))
    img_bitaral = cv2.bilateralFilter(img,d=15,sigmaColor=100,sigmaSpace=100)

    img_blured  = cv2.GaussianBlur(img_bitaral,(11,11),0)

    img_sharpened = cv2.addWeighted(img_bitaral,1.5,img_blured,-0.5,0)

    hsv = cv2.cvtColor(img_sharpened,cv2.COLOR_BGR2HSV)
    hsv[...,1]=hsv[...,1]*1.5
    hsv[...,1]= np.clip(hsv[...,1],0,255).astype(np.uint8)
    img_staturated = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    noise = np.random.normal(scale=5,size=img_staturated.shape).astype(np.uint8)

    img_texutred = cv2.add(img_staturated,noise)

    acrylic_painting = np.clip(img_texutred,0,255).astype(np.uint8)


    output_path = image_path.replace('.jpg','acrylic_painting.jpg')
    cv2.imwrite(output_path,acrylic_painting)
    return output_path    

# pen and ink drawing

def convert_to_pen_and_ink(image_path):
    if not image_path.lower().endswith('.jpg'):
        return None 
    img = cv2.imread(image_path)

    h,w = img.shape[:2]
    # if max(h,w) > 1200:
    #     scale = 1200/max(h,w)
    #     img = cv2.resize(img,(int(h*scale),int(w*scale)))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inverted_gray,(21,21),0)

    sketch =cv2.divide(gray,255,inverted_gray,255)

    _,ink_effect = cv2.threshold(sketch,127,255,cv2.THRESH_BINARY)

    noise = np.random.normal(0,5,gray.shape).astype(np.uint8)

    ink_effect = cv2.add(ink_effect,noise)

    pen_and_ink = np.clip(ink_effect,0,255).astype(np.uint8)

    output_path = image_path.replace('.jpg','pen_and_ink.jpg')
    cv2.imwrite(output_path,pen_and_ink)
    return output_path


# spary painting

def convert_to_spray_painting(image_path):
    if not image_path.lower().endswith('.jpg'):
        return None 
    img = cv2.imread(image_path)
    h,w = img.shape[:2]
    # if max(h,w) > 1200 :
    #     scale = 1200/max(h,w)
    #     img=cv2.resize(img,(int(w*scale),int(h*scale)))
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv[...,1]= cv2.add(hsv[...,1],40)
    img_statured = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    blurred =cv2.GaussianBlur(img_statured,(11,11),3)

    noise = np.random.normal(0,20,img.shape).astype(np.int16)
    noisy_img = np.clip(blurred.astype(np.int16)+noise,0,255).astype(np.uint8)

    splatter = np.zeros((h,w),dtype=np.uint8)
    
    for _ in range(300):
        x,y = np.random.randint(0,w),np.random.randint(0,h)
        radius =np.random.randint(1,5)
        cv2.circle(splatter,(x,y),radius,255,1)

    splatter =cv2.GaussianBlur(splatter,(7,7),3)
    splatter_mask = cv2.cvtColor(splatter,cv2.COLOR_GRAY2BGR)/255.0

    spray_paint = noisy_img.astype(np.float32) * (1- splatter_mask) +splatter_mask *255
    spray_paint = np.clip(spray_paint,0,255).astype(np.uint8)


    output_path = image_path.replace('.jpg','spray_painting.jpg')
    cv2.imwrite(output_path,spray_paint)
    return output_path


# tatoo drawing

def convert_to_tattoo_drawing(image_path):
    if not image_path.lower().endswith('.jpg'):
        return None 
    img=cv2.imread(image_path)
    h,w= img.shape[:2]

    # if max(h,w) > 1200:
    #     scale = 1200/max(h,w)
    #     img = cv2.resize(img,(int(h*scale),int(w*scale)))

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    thredshold_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    
    edges = cv2.Canny(thredshold_img,100,200)

    contrast_img = cv2.convertScaleAbs(edges,alpha=2.0,beta=50)

    kernel =np.ones((3,3),np.uint8)

    delaited_edges = cv2.dilate(contrast_img,kernel,iterations=2)

    blurred = cv2.GaussianBlur(delaited_edges,(3,3),0)

    tatoo_drawing = cv2.addWeighted(contrast_img,0.7,blurred,0.3,0)
    tatoo_drawing = cv2.convertScaleAbs(tatoo_drawing,alpha=1.2,beta=0)

    output_path = image_path.replace('.jpg','tatoo_drawing.jpg')
    cv2.imwrite(output_path,tatoo_drawing)
    return output_path

# hatching drawing

def convert_to_hatching_drawing(image_path):
    if not image_path.lower().endswith('.jpg'):
        return None 
    img = cv2.imread(image_path)
    h,w=img.shape[:2]
    # if max(h,w)>1200:
    #     scale = 1200/max(h,w)
    #     img = cv2.resize(img,(int(w*scale),int(h*scale)))
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)

    hatching_image = np.ones((h,w,3),dtype=np.uint8) *255

    line_spacing = 10 
    max_line_density = 20

    for y in range(0,h,line_spacing):
        for x in range(0,w,line_spacing):
            intensity = gray[y,x]

            if intensity < 150 :
                line_density = int((150 -intensity)/10)
                for i in range(line_density):
                    cv2.line(hatching_image, (x - 2, y + i * 2), (x + 30, y + i * 2), (0, 0, 0), 1)
            if intensity > 100:
                line_density = int((intensity-100)/10)
                for i in range(line_density):
                    angle = random.choice([45,135])
                    length = random.randint(10,20)
                    x_offset = int(math.cos(math.radians(angle)) * length)
                    y_offset = int(math.sin(math.radians(angle)) * length)
                    cv2.line(hatching_image, (x, y), (x + x_offset, y + y_offset), (0, 0, 0), 1)

    hatching_image[edges == 255]=[0,0,0]

    
    output_path = image_path.replace('.jpg','hatching_image.jpg')
    cv2.imwrite(output_path,hatching_image)
    return output_path

# with calligraphy pen  drawing

def convert_to_calligraphy_drawing(image_path):
    brush_size=5
    contrast =1.5
    if not image_path.lower().endswith('.jpg'):
        return None 
    img =cv2.imread(image_path)
    h,w =img.shape[:2]
    # if max(h,w) > 1200:
    #     scale = 1200/max(h,w)
    #     img = cv2.resize(img,(int(h*scale),int(w*scale)))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray,alpha=contrast)

    _,binary =cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    smoothed = cv2. morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

    canvas = np.ones_like(img) *255
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
             if smoothed[y,x]:
                    cv2.line(canvas, (x - brush_size, y), (x + brush_size, y), (0, 0, 0), 1)


    output_path = image_path.replace('.jpg','caligraphy.jpg')
    cv2.imwrite(output_path,canvas)
    return output_path


#  3d drawing


def convert_to_3d_drawing(image_path):
    if not image_path.lower().endswith('.jpg'):
        return None 
    img = cv2.imread(image_path)
    orginal = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    depth_map = cv2.GaussianBlur(gray,(21,21),0)
    depth_map = cv2.normalize(depth_map,None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)

    edges = cv2.Canny(gray,100,200)
    edges = cv2.dilate(edges,None,iterations=1)

    edges_3channels = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)

    edges_3channels = 255-edges_3channels

    detail_enhaced = cv2.detailEnhance(orginal,sigma_s=100,sigma_r=0.15)

    stylized = cv2.stylization(detail_enhaced,sigma_s=60,sigma_r=0.07)


    depth_3channel = cv2.merge([depth_map, depth_map, depth_map])

    highlight = cv2.addWeighted(stylized, 0.7, cv2.GaussianBlur(stylized, (0, 0), 10), 0.3, 0)
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
    
    output_path = image_path.replace('.jpg', '_3d_drawing.jpg')

    cv2.imwrite(output_path, result)
    return output_path


def uploadImage(request):
    sketchurl1 = None   
    sketchurl2 = None  
    sketchurl3 = None  
    sketchurl4 = None  
    sketchurl5 = None  
    sketchurl6 = None  
    sketchurl7 = None  
    sketchurl8 = None  
    sketchurl9 = None  
    sketchurl10 = None  
    sketchurl11 = None  
    sketchurl12 = None  
    sketchurl13 = None  
    sketchurl14 = None  
    original_url=None
    error_message = ''

    if request.method == 'POST':
        form = imageuploadform(request.POST, request.FILES)
        image_file = request.FILES.get('image_file') 
        if form.is_valid():
            upload_image = form.save()
            original_url = upload_image.image.url 
            
            if not upload_image.image.name.lower().endswith('.jpg'):
                error_message = 'Please upload a valid JPG image.'
                return render(request, 'upload.html', {
                    'form': form,
                    'error_message': error_message
                })        
           
            sketch_path1 = convert_to_sketch_edges(upload_image.image.path)  
            sketch_path2 = convert_to_sketch_pencil(upload_image.image.path)  
            sketch_path3 = convert_to_watercolor(upload_image.image.path)  
            sketch_path4 = convert_to_sketch_edges1(upload_image.image.path)  
            sketch_path5 = convert_to_oil_painting(upload_image.image.path)  
            sketch_path6 = convert_to_charcoal(upload_image.image.path)  
            sketch_path7 = convert_to_digital_painting(upload_image.image.path)  
            sketch_path8 = convert_to_acrylic_painting(upload_image.image.path)  
            sketch_path9 = convert_to_pen_and_ink(upload_image.image.path)  
            sketch_path10 = convert_to_spray_painting(upload_image.image.path)  
            sketch_path11 = convert_to_tattoo_drawing(upload_image.image.path)  
            sketch_path12 = convert_to_hatching_drawing(upload_image.image.path)  
            sketch_path13 = convert_to_calligraphy_drawing(upload_image.image.path)  
            sketch_path14 = convert_to_3d_drawing(upload_image.image.path)  
            
            
            sketchurl1 = upload_image.image.url.replace('.jpg', '_edges.jpg')
            sketchurl2 = upload_image.image.url.replace('.jpg', '_edges1.jpg')
            sketchurl3 = upload_image.image.url.replace('.jpg', '_pencil.jpg')
            sketchurl4 = upload_image.image.url.replace('.jpg', '_watercolor.jpg')
            sketchurl5 = upload_image.image.url.replace('.jpg', '_oil_paintig.jpg')
            sketchurl6 = upload_image.image.url.replace('.jpg', 'charcoal.jpg')
            sketchurl7 = upload_image.image.url.replace('.jpg', 'digital_painting.jpg')
            sketchurl8 = upload_image.image.url.replace('.jpg', 'acrylic_painting.jpg')
            sketchurl9 = upload_image.image.url.replace('.jpg', 'pen_and_ink.jpg')
            sketchurl10 = upload_image.image.url.replace('.jpg', 'spray_painting.jpg')
            sketchurl11 = upload_image.image.url.replace('.jpg', 'tatoo_drawing.jpg')
            sketchurl12 = upload_image.image.url.replace('.jpg', 'hatching_image.jpg')
            sketchurl13 = upload_image.image.url.replace('.jpg', 'caligraphy.jpg')
            sketchurl14 = upload_image.image.url.replace('.jpg', '_3d_drawing.jpg')
    
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
        'sketchurl7': sketchurl7,
        'sketchurl8': sketchurl8,
        'sketchurl9': sketchurl9,
        'sketchurl10': sketchurl10,
        'sketchurl11': sketchurl11,
        'sketchurl12': sketchurl12,
        'sketchurl13': sketchurl13,
        'sketchurl14': sketchurl14,
    })


