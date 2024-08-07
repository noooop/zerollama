# Adapted from
# https://github.com/infiniflow/ragflow/blob/main/deepdoc/vision/seeit.py

import PIL
import numpy as np
from PIL import ImageDraw, ImageFont, Image


def imagedraw_textsize_c(draw, text):
    if int(PIL.__version__.split('.')[0]) < 10:
        tw, th = draw.textsize(text)
    else:
        left, top, right, bottom = draw.textbbox((0, 0), text)
        tw, th = right - left, bottom - top

    return tw, th


def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_box(im, result, lables, threshold=0.5):
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    color_list = get_color_map_list(len(lables))
    clsid2color = {n.lower():color_list[i] for i,n in enumerate(lables)}
    result = [r for r in result if r.confidence >= threshold]

    for dt in result:
        color = tuple(clsid2color[dt.label.lower()])
        xmin, ymin, xmax, ymax = dt.bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)

        # draw label
        text = "{} {:.4f}".format(dt.label, dt.confidence)
        tw, th = imagedraw_textsize_c(draw, text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
    return im


def get_annotated_image(image, results):
    return draw_box(image, results.bboxes, results.class_names)



