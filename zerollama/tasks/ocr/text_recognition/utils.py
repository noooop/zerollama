
import copy
from surya.schema import Bbox
from surya.postprocessing.heatmap import draw_polys_on_image


def get_annotated_image(image, prediction):
    polys = [Bbox(bbox=s.bbox).polygon for s in prediction.bboxes]
    return draw_polys_on_image(polys, copy.deepcopy(image))



