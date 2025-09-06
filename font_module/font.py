import yaml
import os
from typing import Dict
import pickle

__all__ = ["load_fonts", "DSFont"]


class DSFont:
    def __init__(self, full_path, relative_file_path, font_name, language):
        self.path = full_path
        self.relative_path = relative_file_path
        self.font_name = font_name
        self.language = language

def get_files(root_path: str):
    full_relative_paths = []
    relative_paths = []
    file_names = []
    font_extensions = (".ttf", ".otf", ".woff2")

    normalized_root = os.path.normpath(root_path)
    for dirpath, _, filenames in os.walk(normalized_root):
        for filename in filenames:
            if not filename.endswith(font_extensions):
                continue
            full_path = os.path.join(dirpath, filename)
            full_relative_paths.append(full_path)

            rel_path_from_root = os.path.relpath(full_path, normalized_root)
            relative_paths.append(rel_path_from_root)

            file_names.append(filename)

    return full_relative_paths, relative_paths, file_names


def load_fonts(config_path="configs/font.yml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ds_config = config["dataset"]
    ds_path = ds_config["path"]

    font_list = []

    for spec in ds_config["specs"]:
        for spec_path in spec["path"]:
            spec_path_combine = os.path.join(ds_path, spec_path)
            full_spec_files, relative_spec_files, spec_files = get_files(spec_path_combine)

            if spec.keys().__contains__("rule"):
                rule = eval(spec["rule"])
            else:
                rule = None

            for fi in range(len(spec_files)):
                if rule is not None and not rule(file):
                    print("skip: " + file)
                    continue
                full_file_path =  full_spec_files[fi]
                relative_file_path = relative_spec_files[fi]
                file_name = os.path.splitext(spec_files[fi])[0]
                font_list.append( DSFont( full_file_path, relative_file_path, file_name, spec["language"]  )    )

    font_list.sort(key=lambda x: x.path)

    exclusion_list = ds_config["exclusion"] if ds_config["exclusion"] != None else []
    exclusion_list = [os.path.join(ds_path, path) for path in exclusion_list]

    def exclusion_rule(font: DSFont):
        for exclusion in exclusion_list:
            if os.path.samefile(font.path, exclusion):
                return True
        return False

    return font_list, exclusion_rule


def load_font_with_exclusion(
    config_path="configs/font.yml", cache_path="font_list_cache.bin"
) -> Dict:
    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))
    font_list, exclusion_rule = load_fonts(config_path)
    font_list = list(filter(lambda x: not exclusion_rule(x), font_list))
    font_list.sort(key=lambda x: x.path)
    print("font count: " + str(len(font_list)))
    ret = {font_list[i].path: i for i in range(len(font_list))}
    with open(cache_path, "wb") as f:
        pickle.dump(ret, f)
    return ret
