import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10));
from matplotlib import rcParams
import scipy.ndimage

from skimage.measure import label, regionprops
import numpy as np

def name(obj, callingLocals=locals()):
    for k, v in list(callingLocals.items()):
         if v is obj:
            nm = k
    print(nm)

def resize_(image):
    h, w = image.shape[:2]
    if h<400:
        dif = int((400 - h)/2)
        pad_top = dif
        pad_bot = dif
        
        if pad_top + dif + h != 399:
            pad_top = dif + 1 
        
        image = cv2.copyMakeBorder(image, pad_top, pad_bot, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
    return image    
    
def plot_(n,text_='',plotVisible=True):
    
    if plotVisible==True:
        print(text_)
        plt.imshow(n, cmap='gray')
        o = plt.show()
    return 0


def plotX(numplots=0, img1=False,label1="",img2=False,label2="",img3=False,label3="",img4=False,label4="", plotVisible=True):
    # Display result
    if plotVisible==True:
        fig, ax_arr = plt.subplots(1, numplots, constrained_layout=False, sharex=False, sharey=False, figsize=(10, 10))
        if numplots==2:
            ax1, ax2 = ax_arr.ravel()

            ax1.imshow(img1, cmap='gray')
            ax1.set_title(label1)

            ax2.imshow(img2, cmap='gray')
            ax2.set_title(label2)

        if numplots==4:
            ax1, ax2, ax3 ,ax4  = ax_arr.ravel()

            ax1.imshow(img1, cmap='gray')
            ax1.set_title(label1)

            ax2.imshow(img2, cmap='gray')
            ax2.set_title(label2)

            ax3.imshow(img3, cmap='gray')
            ax3.set_title(label1)

            ax4.imshow(img4, cmap='gray')
            ax4.set_title(label4)


        for ax in ax_arr.ravel():
            ax.set_axis_off()

        plt.tight_layout()
        plt.show()


    
    
def cut_edges(im_in):
    # Mask used to flood filling.
    ret,mask = cv2.threshold(im_in ,220,255,cv2.THRESH_BINARY)
    mask1 = mask.copy()
    
#     plot_(mask1)
    # Notice the size needs to be 2 pixels than the image.
    h, w = mask1.shape[:2]
    mask_ = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    if mask1[0,0] ==     255:cv2.floodFill(mask1, mask_, (0,0), 0)
    if mask1[h-2, 0] ==  255:cv2.floodFill(mask1, mask_, (0,h-2 ), 0)
    if mask1[0,w-2] ==   255:cv2.floodFill(mask1, mask_, (w-2,0), 0)
    if mask1[h-2,w-2] == 255:cv2.floodFill(mask1, mask_, (w-2,h-2), 0)

    im_out =  (im_in * (~(mask - mask1)/255)).astype('uint8')
#     print(im_out)
#     plot_(im_out)
    return im_out

def remove_holes(im_in):
    
    im_floodfill = im_in.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask_ = np.zeros((h+2, w+2), np.uint8)

    
    print(type(im_floodfill))
    print(np.min(im_floodfill),'-',np.max(im_floodfill))
    print(im_floodfill.shape)
    plt.imshow(im_floodfill, cmap='gray')
    plt.show()

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask_, (0,0), 255);
    cv2.floodFill(im_floodfill, mask_, (w-1,h-1), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv
    
    return im_out


def remove_areas(m):
    L = label(m)
    
    while np.max(L)!=1:
#         print(np.max(L))
        reg_max=0
        reg_max_l = 0
        for region  in (regionprops(L)):
            # take regions with large enough areas
            if region.area > reg_max:
                reg_max= region.area
                reg_max_l = region.label

        L[L!=reg_max_l]=0
        L[L==reg_max_l]=1
        
    return L

def remove_bottom_areas(m,n):
    
    return L

def crop_image(mask, image):
    mask_=mask*1
    len_= mask.shape[0]-1
    y, y_ = 0, 0
    for pos,v in enumerate(mask_):
        for j, v_ in enumerate(v): 
            if v_==1:
                y= pos
                break
                
    for i in range(len(mask_)):
        for j, v_ in enumerate( mask_[len_-i]): 
            if v_==1:
                y_= len_-i
                break


    len_= mask_.shape[1]-1
    mask_=cv2.rotate(mask_, cv2.ROTATE_90_CLOCKWISE)
    
    x, x_ = 0, 0
    for pos,v in enumerate(mask_):
        for j, v_ in enumerate(v): 
            if v_==1:
                x= pos
                break
                
    for i in range(len(mask_)):
        for j, v_ in enumerate( mask_[len_-i]): 
            if v_==1:
                x_= len_-i
                break
    
    newim=image[y_:y,x_:x]
    
    return newim

def get_boundaries(img_bc, boundaries = 0):
    img_bc[img_bc>=1]=1
    y, x = img_bc.shape
    img_fin_bin = np.empty_like (img_bc[:])

    for j in range(5):
        i = j
        img_top_bin = np.empty_like (img_bc[i:])
        img_top_bin[:] = img_bc[i:]

        img_bot_bin = np.empty_like (img_bc[:y-i])
        img_bot_bin[:] = img_bc[:y-i] 

        img_fin_bin[i:] = img_fin_bin[i:] + img_bot_bin
        img_fin_bin[:y-i] = img_fin_bin[:y-i] + img_top_bin


    img_bc[img_fin_bin>=1]=1
    return img_bc

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def normal_(result,t):
    result[result<t]=0  
        
    return result.astype('uint8')

def buf_(result,t):
    result[result>=t]=255
    result[result<t]=0
    
    return result.astype('uint8')

def buf2_(result,t):
    result[result<t]=0
    
    return result.astype('uint8')

def grayscale(rgb):
    gray_ = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    return gray_

def cut_borders(gray_):
    y,x = gray_.shape
    margin = 10
    gray_=gray_[margin:y-margin,margin:x-margin]
    y,x = gray_.shape
    midX = int(x/2) 
    gray_=gray_[:,midX-200:midX+200]

    
    return gray_

def main(filename,original):
#     temp= readImg(filename)
#     gray= grayscale(temp)
#     gray= grayscale(filename)
    inverted= invert(filename)
    b = scipy.ndimage.filters.gaussian_filter(inverted,sigma=5)
    return draw(b,filename,10,original)