import sys, re, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imagehash
from scipy import ndimage
global_font = "Inconsolata-Bold.ttf"


def get_image(image_path):
  with Image.open(image_path) as image:
    im_arr = np.frombuffer(image.tobytes(), dtype=np.uint8)
    im_arr = np.copy(im_arr)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    im_arr = im_arr[:,:,0]
    im_arr[im_arr<100] = 0
  return np.array(im_arr, dtype=bool)

def make_seperate_img(np_img, box, label, extend = True):
    if extend:
        np_img = np.copy(np_img[box[1]:box[3]+2,box[0]:box[2]+2])
        np_img[np_img!=label]=0
        np_img[np_img>0]=1
        return  np_img
    else:
        np_img = np.copy(np_img[box[1]:box[3],box[0]:box[2]])
        return  np_img==label
    return  np_img==label

image_path = "m_test_find_font.jpeg"

image = get_image(image_path)
img = Image.open(image_path)
font = ImageFont.truetype(global_font, 20)

label_img, num_features = ndimage.label(image, structure = [[0,1,0],[1,1,1],[0,1,0]])
bounding_box = ndimage.find_objects(label_img)



for g in range(1,num_features+1):
    box = [ bounding_box[g-1][1].start, bounding_box[g-1][0].start, bounding_box[g-1][1].stop, bounding_box[g-1][0].stop]
    draw = ImageDraw.Draw(img)
    draw.text(( box[0],box[1] ), str(g), (255), font = font)

# img.show()


text_result = "17 Q,29 u,25 a,26 n,20 b,27 a,19 F,21 R,23 E,18 Y"
text_result = [{"id":int(a.split(" ")[0]),"text":a.split(" ")[1]} for a in text_result.split(",")]

for t in text_result:
	g = t["id"]
	box = [ bounding_box[g-1][1].start, bounding_box[g-1][0].start, bounding_box[g-1][1].stop, bounding_box[g-1][0].stop]
	t["hash"] = imagehash.phash( Image.fromarray(make_seperate_img(label_img,box,g) ) )

# print(text_result)

with open('font_hash.json', 'r', encoding="utf-8") as f:
    font_data =  json.load(f)

font_result = []

for f in font_data.keys():
	score = 0 
	for t in text_result:
		score+=  imagehash.hex_to_hash( font_data[f][ t["text"] ] ) - t["hash"]
	font_result+=[{"font":f,"score":score}]

font_result = sorted(font_result, key=lambda d: d['score'])

font_result = font_result[:20]
print(font_result)
