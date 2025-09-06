__all__ = ["FontLabel"]

from typing import Tuple
from .font import DSFont


class FontLabel:
    """
    :param image_width: image width
    :param image_height: image height
    :param text: text
    :param font: font
    :param text_color: text color
    :param text_size: text size
    :param text_direction: text direction, ltr or ttb
    :param stroke_width: stroke width
    :param stroke_color: stroke color
    :param line_spacing: line spacing
    :param language: language
    :param bbox: bounding box, (left, top, width, height)
    :param angle: angle in degrees
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        text: str,
        font: DSFont,
        text_color: Tuple[int, int, int],
        text_size: int,
        text_direction: str,
        stroke_width: int,
        stroke_color: Tuple[int, int, int],
        line_spacing: int,
        language: str,
        bbox: Tuple[int, int, int, int],
        angle: int,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.text = text
        self.font = font
        self.text_color = text_color
        self.text_size = text_size
        self.text_direction = text_direction
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.line_spacing = line_spacing
        self.language = language
        self.bbox = bbox
        self.angle = angle

    def todict(self):
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "text": self.text,
            "font": self.font.font_name,
            "relative_path": self.font.relative_path,
            "full_path": self.font.path,
            "text_color": self.text_color,
            "text_size": self.text_size,
            "text_direction": self.text_direction,
            "stroke_width": self.stroke_width,
            "stroke_color": self.stroke_color,
            "line_spacing": self.line_spacing,
            "language": self.language,
            "bbox": self.bbox,
            "angle": self.angle,
        }