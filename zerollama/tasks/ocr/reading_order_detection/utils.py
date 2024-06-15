
import numpy as np
import supervision as sv


def get_annotated_image(image, results):
    class_names = np.array([str(x.position) for x in results.bboxes])
    confidence = np.array([1.0 for x in results.bboxes])
    xyxy = np.array([x.bbox for x in results.bboxes])
    class_id = np.array([x.position for x in results.bboxes])

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



