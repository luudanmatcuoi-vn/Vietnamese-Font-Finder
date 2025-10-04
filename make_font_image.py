from os import listdir
from os.path import isfile, join, isdir
import sys, re, json, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import PIL.ImageOps
import imagehash
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = 9696447580

FONT_ROOT_PATH = "F:\\Animesoftsub\\vietnamese-font"
font_images_folder = "D:\\Vietnamese-Font-Finder\\font_images"

listfile = [f for f in listdir(font_images_folder) if isfile(join(font_images_folder, f) )]

fonts = []
for path, subdirs, files in os.walk(FONT_ROOT_PATH):
    for name in files:
        if any([a in name.lower() for a in [".ttf","otf"]]) :
            full_path = os.path.join(path,name)
            relative_path = full_path.replace(FONT_ROOT_PATH,"")[1:]
            fonts +=[ [full_path, relative_path, name ] ]

fonts = sorted(fonts)

# print(fonts)

characters_req = [chr(g) for g in range(97,123)]+[chr(ga) for ga in range(65,91)]+list("0123456789")+["ư","đ","Đ","ơ","Ư","Ơ",".",",","?","-"]

def show(im):
    if im.ndim==2:
        im = im.reshape((im.shape[0], im.shape[1], 1))
        im = np.concatenate((im, im,im), axis=2)
    show_img = Image.fromarray(np.uint8(im), 'RGB')
    show_img.show()

def finding_font_size(require_height, global_font, seperator, finding=True,save=False):
    if save:
        img  = Image.new('L', ( 5000, require_height*5 ))
    else:
        img  = Image.new('L', ( 1000, require_height*3 ))
    draw = ImageDraw.Draw(img)

    if finding:
        font_size = 40
        font = ImageFont.truetype(global_font, font_size)
        (l,top,r,bottom) = draw.textbbox((10, 10), "oa", font=font)
        def a_(l,top,r,bottom):
            if bottom-top==0:return r-l
            else: return bottom-top
        while abs( a_(l,top,r,bottom) - require_height)>2:
            if a_(l,top,r,bottom) - require_height>0:
                font_size -= 1
                font = ImageFont.truetype(global_font, font_size)
                (l,top,r,bottom) = draw.textbbox((10, 10), "oa", font=font)
            else:
                font_size += 1
                font = ImageFont.truetype(global_font, font_size)
                (l,top,r,bottom) = draw.textbbox((10, 10), "oa", font=font)
        (left,top,right,bottom) = draw.textbbox((10, 10), seperator, font=font)
    else:
        font_size = require_height
        font = ImageFont.truetype(global_font, font_size)

    # Making np_img
    try:
        if save:
            draw.text(( 100,100 ), seperator, (255), font = font, align ="left")
        else:
            draw.text(( 10,10 ), seperator, (255), font = font)
    except:
        return font_size, "0000000000000000"
    np_img = np.asarray(img)

    if finding:
        width_list = sorted( set(np.where(np_img >100 )[1]) )
        del np_img, img
        if len(width_list)==0:
            delta_width = 0
        else: 
            delta_width = width_list[-1]-width_list[0]
    else:
        bounding_box = ndimage.find_objects(np_img)
        if save:
            widthl = sorted( set(np.where(np_img >20 )[1]) )
            heightl = sorted( set(np.where(np_img >20 )[0]) )
            try:
                aimg = img.crop((widthl[0]-10,heightl[0]-10,widthl[-1]+20,heightl[-1]+10))
                aimg = ImageOps.invert(aimg)
                aimg.save(save + ".jpg")
            except:
                pass
        try:
            img = img.crop((bounding_box[-1][1].start, bounding_box[-1][0].start, bounding_box[-1][1].stop, bounding_box[-1][0].stop))
            delta_width = str( imagehash.phash( img ) )
        except:
            delta_width = "0000000000000000"

    return font_size, delta_width

def create_map(font, characters_req):
    font_size,_ = finding_font_size(30, font[0],"o",finding =True)
    result_map = {}
    for ch in characters_req:
        a, wid = finding_font_size(font_size, font[0], ch, finding=False)
        result_map[ch] = wid
    return result_map
    
multi_data = {}
try:
    with open('font_hash.json', 'r', encoding="utf-8") as f:
        multi_data = json.load(f)
except:
    pass

##Thêm các font mới vào font_hash
for font in fonts:
    if font[1] in multi_data.keys(): continue
    print("add font", font)
    data = create_map(font, characters_req)
    multi_data[font[1]] = data
    with open('font_hash.json', 'w', encoding="utf-8") as f:
        json.dump(multi_data, f)



##Xoá các font ko có trong host
clannad = [font[1] for font in fonts]
multi_data_2 = {}
for k in multi_data.keys():
    if k in clannad:
        multi_data_2[k] = multi_data[k]
    else:
        print("remove font", multi_data[k])
with open('font_hash.json', 'w', encoding="utf-8") as f:
    json.dump(multi_data_2, f)



##Thêm ảnh cho các font mới
for font in fonts:
    if font[2]+".jpg" in listfile: continue
    print("add image", font)
    font_size,_ = finding_font_size(80, font[0],"o",finding =True)
    _,_ = finding_font_size(font_size, font[0], "Phông Chữ Mẫu\"?!,.", finding=False, save=font_images_folder+"\\"+font[2])


##Xoá ảnh font mà ko có host
clannad = [ font[2]+".jpg" for font in fonts]
for g in listfile:
    if g not in clannad:
        print("remove image", g)
        os.remove( os.path.join(font_images_folder, g) )