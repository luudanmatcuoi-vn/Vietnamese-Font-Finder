import matplotlib.font_manager

import sys, re, json, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL.ImageOps
import imagehash
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = 9696447580

FONT_ROOT_PATH = "D:\\rank\\vietnamese-font"
fonts = []
for path, subdirs, files in os.walk(FONT_ROOT_PATH):
    for name in files:
        if any([a in name.lower() for a in [".ttf","otf"]]) :
            full_path = os.path.join(path,name)
            relative_path = full_path.replace(FONT_ROOT_PATH,"")[1:]
            fonts +=[ [full_path, relative_path, name ] ]

forbidden_fonts = ["MTD Authemart Extras"]
if "fonts" not in globals():
    fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    fonts = [f for f in fonts if any([a.lower() in f.lower() for a in ["uvn","utm"]])]
    fonts = [f for f in fonts if not any([a.lower() in f.lower() for a in forbidden_fonts])]
    fonts = sorted(list(set(fonts)))
    fonts = [[f, f.split("\\")[-1], f.split("\\")[-1] ] for f in fonts]

# f=0
# while f <len(fonts):
#     fa = [a for a in fonts if a[1]==fonts[f][1] ][1:]
#     for a in fa:
#         fonts.remove(a)
#     f+=1

fonts = sorted(fonts)

# print(fonts)

characters_req = [chr(g) for g in range(97,123)]+[chr(ga) for ga in range(65,91)]+list("0123456789")+["ư","đ","Đ","ơ","Ư","Ơ",".",",","?","-"]

def show(im):
    if im.ndim==2:
        im = im.reshape((im.shape[0], im.shape[1], 1))
        im = np.concatenate((im, im,im), axis=2)
    show_img = Image.fromarray(np.uint8(im), 'RGB')
    show_img.show()

def finding_font_size(require_height, global_font, seperator, finding=True):
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
        try:
            img = img.crop((bounding_box[-1][1].start, bounding_box[-1][0].start, bounding_box[-1][1].stop, bounding_box[-1][0].stop))
            delta_width = str( imagehash.phash( img ) )
        except:
            delta_width = "0000000000000000"

    return font_size, delta_width

def create_map(font, characters_req):
    font_size,_ = finding_font_size(80, font[0],"o",finding =True)
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

for font in fonts:
    if font[1] in multi_data.keys(): continue
    print(font)
    data = create_map(font, characters_req)
    multi_data[font[1]] = data
    with open('font_hash.json', 'w', encoding="utf-8") as f:
        json.dump(multi_data, f)

# print(multi_data)

