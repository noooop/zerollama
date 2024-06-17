# Adapted from
# https://github.com/VikParuchuri/surya/blob/master/surya/postprocessing/heatmap.py

import copy
import platform
from PIL import ImageDraw, ImageFont, Image


def get_font_path():
    plat = platform.system().lower()
    if plat == 'windows':
        return "msyh.ttc"
    else:
        # sudo apt-get install fonts-wqy-microhei ttf-wqy-zenhei
        return "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


def get_text_size(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height


def draw_polys_on_image(bboxes, image, labels=None, box_padding=-1, label_offset=1, label_font_size=10):
    corners = [s.polygon for s in bboxes]
    draw = ImageDraw.Draw(image)
    font_path = get_font_path()
    label_font = ImageFont.truetype(font_path, label_font_size)

    for i in range(len(corners)):
        poly = corners[i]
        poly = [(int(p[0]), int(p[1])) for p in poly]
        draw.polygon(poly, outline='red', width=1)

        if labels is not None:
            label = labels[i]
            text_position = (
                min([p[0] for p in poly]) + label_offset,
                min([p[1] for p in poly]) + label_offset
            )
            text_size = get_text_size(label, label_font)
            box_position = (
                text_position[0] - box_padding + label_offset,
                text_position[1] - box_padding + label_offset,
                text_position[0] + text_size[0] + box_padding + label_offset,
                text_position[1] + text_size[1] + box_padding + label_offset
            )
            draw.rectangle(box_position, fill="white")
            draw.text(
                text_position,
                label,
                fill="red",
                font=label_font
            )

    return image


def get_annotated_image(image, prediction):
    return draw_polys_on_image(prediction.bboxes, copy.deepcopy(image))



