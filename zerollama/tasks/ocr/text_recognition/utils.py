# Adapted from
# https://github.com/VikParuchuri/surya/blob/master/surya/postprocessing/text.py

import re
from PIL import ImageDraw, ImageFont, Image
from zerollama.tasks.ocr.text_line_detection.utils import get_font_path, get_text_size


def render_text(draw, text, s_bbox, bbox_width, bbox_height, font_path, box_font_size):
    font = ImageFont.truetype(font_path, box_font_size)
    text_width, text_height = get_text_size(text, font)
    while (text_width > bbox_width or text_height > bbox_height) and box_font_size > 6:
        box_font_size = box_font_size - 1
        font = ImageFont.truetype(font_path, box_font_size)
        text_width, text_height = get_text_size(text, font)

    # Calculate text position (centered in bbox)
    text_width, text_height = get_text_size(text, font)
    x = s_bbox[0]
    y = s_bbox[1] + (bbox_height - text_height) / 2

    draw.text((x, y), text, fill="black", font=font)


def render_math(image, draw, text, s_bbox, bbox_width, bbox_height, font_path):
    try:
        from surya.postprocessing.math.render import latex_to_pil
        box_font_size = max(10, min(int(.2 * bbox_height), 24))
        img = latex_to_pil(text, bbox_width, bbox_height, fontsize=box_font_size)
        img.thumbnail((bbox_width, bbox_height))
        image.paste(img, (s_bbox[0], s_bbox[1]))
    except Exception as e:
        print(f"Failed to render math: {e}")
        box_font_size = max(10, min(int(.75 * bbox_height), 24))
        render_text(draw, text, s_bbox, bbox_width, bbox_height, font_path, box_font_size)


def is_latex(text):
    latex_patterns = [
        r'\\(?:begin|end)\{[a-zA-Z]*\}',
        r'\$.*?\$',
        r'\$\$.*?\$\$',
        r'\\[a-zA-Z]+',
        r'\\[^a-zA-Z]',
    ]

    combined_pattern = '|'.join(latex_patterns)
    if re.search(combined_pattern, text, re.DOTALL):
        return True

    return False


def draw_text_on_image(bboxes, texts, image_size, max_font_size=60, res_upscale=2, has_math=False):
    font_path = get_font_path()
    new_image_size = (image_size[0] * res_upscale, image_size[1] * res_upscale)
    image = Image.new('RGB', new_image_size, color='white')
    draw = ImageDraw.Draw(image)

    for bbox, text in zip(bboxes, texts):
        s_bbox = [int(coord * res_upscale) for coord in bbox]
        bbox_width = s_bbox[2] - s_bbox[0]
        bbox_height = s_bbox[3] - s_bbox[1]

        # Shrink the text to fit in the bbox if needed
        if has_math and is_latex(text):
            render_math(image, draw, text, s_bbox, bbox_width, bbox_height, font_path)
        else:
            box_font_size = max(6, min(int(.75 * bbox_height), max_font_size))
            render_text(draw, text, s_bbox, bbox_width, bbox_height, font_path, box_font_size)

    return image


def get_annotated_image(image, prediction, langs):
    bboxes = [l.bbox for l in prediction.text_lines]
    text = [l.text for l in prediction.text_lines]
    return draw_text_on_image(bboxes, text, image.size, has_math="_math" in langs)



