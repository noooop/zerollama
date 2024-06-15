
from surya.postprocessing.text import draw_text_on_image


font_path = "arial.ttf"


def get_annotated_image(image, prediction, langs):
    bboxes = [l.bbox for l in prediction.text_lines]
    text = [l.text for l in prediction.text_lines]
    return draw_text_on_image(bboxes, text, image.size, langs, has_math="_math" in langs, font_path=font_path)



