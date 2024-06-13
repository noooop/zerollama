
import numpy as np
import supervision as sv


def get_annotated_image(image, results):
    class_id = np.array([x['class'] for x in results])
    class_names = np.array([x['name'] for x in results])
    confidence = np.array([x['confidence'] for x in results])
    xyxy = np.array([[x["box"]["x1"], x["box"]["y1"], x["box"]["x2"], x["box"]["y2"]] for x in results])

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data={"class_name": class_names},
    )

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    return annotated_image



