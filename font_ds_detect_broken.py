import sys
import traceback
import pickle
import os
import concurrent.futures
from tqdm import tqdm
import time
from font_module.font import load_fonts
import cv2
import json

cjk_ratio = 3
img_per_font = 10

dataset_path = "./dataset/font_img"
os.makedirs(dataset_path, exist_ok=True)

unqualified_log_file_name = f"unqualified_font_{time.time()}.txt"
runtime_exclusion_list = []

fonts, exclusion_rule = load_fonts()


def generate_dataset(dataset_type: str, cnt: int):
    if dataset_type =="":
        dataset_bath_dir = dataset_path
    else:
        dataset_bath_dir = os.path.join(dataset_path, dataset_type)
    os.makedirs(dataset_bath_dir, exist_ok=True)

    def _generate_single(args):
        i, j, font = args
        # print(
        #     f"Checking {dataset_type} font: {font.path} {i} / {len(fonts)}, image {j}",
        #     end="\r",
        # )

        if exclusion_rule(font):
            print(f"Excluded font: {font.path}")
            return
        if font.path in runtime_exclusion_list:
            print(f"Excluded font: {font.path}")
            return

        image_file_name = f"font_{i}_img_{j}.jpg"
        label_file_name = f"font_{i}_img_{j}.json"

        image_file_path = os.path.join(dataset_bath_dir, image_file_name)
        label_file_path = os.path.join(dataset_bath_dir, label_file_name)

        # detect cache
        if (not os.path.exists(image_file_path)) or (
            not os.path.exists(label_file_path)
        ):
            print(
                f"Missing {dataset_type} font: {font.path} {i} / {len(fonts)}, image {j}"
            )
            try:
                os.remove(image_file_path)
                os.remove(label_file_path)
            except:
                pass
        else:

            # detect broken
            try:
                # check image
                cv2.imread(image_file_path)
                # # check label
                with open(label_file_path, "r") as f:
                    baka = json.load(f)
                if "z-font-chua-viet-hoa-hoac-viet-hoa-mot-phan-hoac-viet-hoa-ko-phai-unicode" in baka["full_path"]:
                    raise ValueError("Bakadayo")
            except Exception as e:
                print(
                    f"Broken {dataset_type} font: {font.path} {i} / {len(fonts)}, image {j}"
                )
                os.remove(image_file_path)
                os.remove(label_file_path)
                # exit()

        return

    work_list = []

    # divide len(fonts) into 64 parts and choose the third part for this script
    for i in range(len(fonts)):
        font = fonts[i]
        if font.language == "CJK":
            true_cnt = cnt * cjk_ratio
        else:
            true_cnt = cnt
        for j in range(true_cnt):
            work_list.append((i, j, font))

    for i in tqdm(range(len(work_list))):
        _generate_single(work_list[i])


generate_dataset("", img_per_font)
