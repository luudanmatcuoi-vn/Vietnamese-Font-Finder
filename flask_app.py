from flask import Flask, render_template, request, jsonify
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageOps as ImageOps
import imagehash
from scipy import ndimage
import json
import io
import base64
import os
import urllib.parse
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def get_image_array(image_data, threshold=100, invert=False):
    """Convert image to binary array with threshold and invert options"""
    if isinstance(image_data, str):
        # If it's a base64 string, decode it
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
    else:
        # If it's already a PIL Image
        img = image_data
    
    img = img.convert('L')
    im_arr = np.array(img)
    
    # Apply threshold
    im_arr[im_arr < threshold] = 0
    im_arr[im_arr >= threshold] = 255

    # Invert if requested
    if invert:
        im_arr = 255 - im_arr
    
    # Convert to boolean
    binary_arr = np.array(im_arr, dtype=bool)
    
    return binary_arr, im_arr

def process_image_labels(image_array):
    """Process image to find connected components and bounding boxes"""
    label_img, num_features = ndimage.label(image_array, structure=[[0,1,0],[1,1,1],[0,1,0]])
    bounding_boxes = ndimage.find_objects(label_img)
    return label_img, num_features, bounding_boxes

def make_separate_img(np_img, box, label):
    """Extract separated character image"""
    np_img = np.copy(np_img[box[1]:box[3], box[0]:box[2]])
    # if not isinstance(label, int):
    #     for g in label:
    #         np_img[np_img == g] = 500
    #     label = 500
    np_img[np_img != label] = 0
    np_img[np_img > 0] = 1
    return np_img


def get_click_label(x, y, label_img):
    """Get label ID at clicked coordinates"""
    if 0 <= y < label_img.shape[1] and 0 <= x < label_img.shape[1]:
        return int(label_img[y, x])
    return 0

def numpy_to_base64(np_array):
    """Convert numpy array to base64 string"""
    # Convert to uint8 if it's not already
    if np_array.dtype != np.uint8:
        np_array = np_array.astype(np.uint8)
    
    img = Image.fromarray(np_array, 'L')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def base64_to_numpy(base64_str):
    """Convert base64 string to numpy array"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        threshold = int(request.form.get('threshold', 100))
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process the uploaded image
        img = Image.open(file.stream)
        
        # Process with threshold (normal)
        image_array, processed_array = get_image_array(img, threshold, False)
        label_img, num_features, bounding_boxes = process_image_labels(image_array)
        
        # Process with threshold (inverted)
        image_array_invert, processed_array_invert = get_image_array(img, threshold, True)
        label_img_invert, num_features_invert, bounding_boxes_invert = process_image_labels(image_array_invert)
        
        # Convert to base64
        processed_b64 = numpy_to_base64(processed_array)
        processed_invert_b64 = numpy_to_base64(processed_array_invert)
        label_b64 = numpy_to_base64(label_img.astype(np.uint8))
        label_invert_b64 = numpy_to_base64(label_img_invert.astype(np.uint8))
        
        # Convert bounding boxes to serializable format
        def serialize_bounding_boxes(boxes):
            serialized = {}
            idd=0
            while idd<len(boxes):
                serialized[idd+1] = {
                    'start_0': int(boxes[idd][0].start),
                    'stop_0': int(boxes[idd][0].stop),
                    'start_1': int(boxes[idd][1].start),
                    'stop_1': int(boxes[idd][1].stop)
                }
                idd+=1
            return serialized
        
        return jsonify({
            'success': True,
            'processed_image': processed_b64,
            'processed_image_invert': processed_invert_b64,
            'label_image': label_b64,
            'label_image_invert': label_invert_b64,
            'num_features': int(num_features),
            'num_features_invert': int(num_features_invert),
            'bounding_boxes': serialize_bounding_boxes(bounding_boxes),
            'bounding_boxes_invert': serialize_bounding_boxes(bounding_boxes_invert)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_label', methods=['POST'])
def get_label():
    try:
        data = request.json
        x = int(data.get('x'))
        y = int(data.get('y'))
        label_image_b64 = data.get('label_image')
        
        if not label_image_b64:
            return jsonify({'error': 'Label image not provided'}), 400
        
        # Convert base64 to numpy array
        label_img = base64_to_numpy(label_image_b64)
        
        # Get label at coordinates
        label_id = get_click_label(x, y, label_img)
        
        return jsonify({
            'label_id': label_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/find_fonts', methods=['POST'])
def find_fonts():
    try:
        data = request.json
        char_mappings = data.get('characters', [])
        label_image_b64 = data.get('label_image')
        bounding_boxes_data = data.get('bounding_boxes', [])
        
        if not char_mappings:
            return jsonify({'error': 'No character mappings provided'}), 400
        
        if not label_image_b64:
            return jsonify({'error': 'Label image not provided'}), 400
        
        # Convert base64 label image to numpy array
        label_img = base64_to_numpy(label_image_b64)

        
        # Reconstruct bounding boxes from serialized format
        def deserialize_bounding_boxes(serialized_boxes):
            boxes = []
            for bb in serialized_boxes.keys():
                box_data = serialized_boxes[bb]
                if box_data is not None:
                    box = (
                        slice(box_data['start_0'], box_data['stop_0']),
                        slice(box_data['start_1'], box_data['stop_1'])
                    )
                    boxes.append(box)
                else:
                    boxes.append(None)
            return boxes
        
        bounding_boxes = deserialize_bounding_boxes(bounding_boxes_data)
        
        # Create text_result format
        text_result = []
        for mapping in char_mappings:
            if mapping['letter'] and mapping['label_id'] > 0:
                text_result.append({
                    'text': mapping['letter'].upper(),
                    'id': mapping['label_id']
                })
        
        # Calculate hashes for each character
        for t in text_result:
            label_id = t['id']

            if label_id <= len(bounding_boxes) and bounding_boxes[label_id - 1] is not None:
                bbox = bounding_boxes[label_id - 1]
                box = [bbox[1].start, bbox[0].start, bbox[1].stop, bbox[0].stop]
            
            if box:
                # Extract character image
                temp_np_img = make_separate_img(label_img, box, label_id)
                
                # Calculate perceptual hash
                char_img = Image.fromarray(temp_np_img.astype(np.uint8) * 255)
                t['hash'] = imagehash.phash(char_img)
        
        # Load font database
        font_data_path = 'font_hash.json'
        try:
            with open(font_data_path, 'r', encoding='utf-8') as f:
                font_data = json.load(f)
        except FileNotFoundError:
            return jsonify({'error': 'Font database not found'}), 500
        
        # Calculate font similarities
        font_results = []
        for font_name in font_data.keys():
            score = 0
            
            for char_result in text_result:
                char_text = char_result['text']
                if char_text in font_data[font_name]:
                    try:
                        font_hash = imagehash.hex_to_hash(font_data[font_name][char_text])
                        score += abs(font_hash - char_result['hash'])
                    except:
                        continue

            font_results.append({
                'font': font_name,
                'score': float(score),
            })
        
        # Sort by score (lower is better)
        font_results = sorted(font_results, key=lambda x: x['score'])[:25]
        
        # Add image URLs
        for result in font_results:
            encoded_name = urllib.parse.quote(result['font'])
            image_url = "https://raw.githubusercontent.com/luudanmatcuoi-vn/Vietnamese-Font-Finder/main/font_images/" + urllib.parse.quote(result["font"].split("\\")[-1]) + ".jpg?raw=1"
            download_url = result["font"].replace('\\', '/')
            download_url = "https://git.linuxholic.com/boydaihungst/vietnamese-font/src/branch/master/" + download_url
            result['image_url'] = image_url
            result['download_url'] = download_url
        
        return jsonify({
            'success': True,
            'fonts': font_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)