import streamlit as st
import numpy as np
import cv2
from PIL import Image

import tensorflow as tf
import torch
import easyocr

from object_detection.utils import config_util, label_map_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils


# ------------------------ Configurations ------------------------

CONFIG_PATH = "./deploy/pipeline.config"
CKPT_PATH = "./deploy/ckpt-14"
LABEL_MAP_PATH = "./deploy/label_map.pbtxt"
DETECTION_THRESHOLD = 0.75
OCR_THRESHOLD = 0.01

# ----------------------------------------------------------------


def object_detector(image):
    """
    Highlight car plate and extract plate number from image
    :param      image: Input numpy image
    :return:    cpn  : Car plate number
                roi  : Original image with bounding box
    """
    config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(model_config=config["model"], is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(CKPT_PATH).expect_partial()

    @tf.function
    def detect_fn(img):
        img, shapes = detection_model.preprocess(img)
        prediction_dict = detection_model.predict(img, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["num_detections"] = num_detections

    # detection_classes should be ints
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"],
        detections["detection_classes"] + label_id_offset,
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=DETECTION_THRESHOLD,
        agnostic_mode=False
    )

    return image_np_with_detections, detections


def extract_ROIs(image, detections):
    height, width, _ = image.shape
    ROIs = [box for box, score in zip(detections["detection_boxes"], detections["detection_scores"]) if
            score > DETECTION_THRESHOLD]
    if len(ROIs) == 0:
        return None
    result_ROIs = []
    for ROI in ROIs:
        absolute_ROI = ROI * [height, width, height, width]
        absolute_ROI = [int(coord) for coord in absolute_ROI]
        absolute_ROI = image[absolute_ROI[0]:absolute_ROI[2], absolute_ROI[1]:absolute_ROI[3], :]
        result_ROIs.append(absolute_ROI)

    return result_ROIs


def apply_OCR(ROIs):
    # Apply OCR to this ROI
    reader = easyocr.Reader(['en', 'de'], gpu=True)

    for ROI in ROIs:
        result = reader.readtext(ROI)

        # Calculate area of detected regions, take the one greater than percent of total area based on ROI_AREA_RATIO_THRESHOLD
        # Coordinates go in clock-wise direction, start from top left corner
        ROI_area = ROI.shape[0] * ROI.shape[1]
        text = []
        for detected_region in result:
            tl = np.array(detected_region[0][0])
            tr = np.array(detected_region[0][1])
            bl = np.array(detected_region[0][3])
            current_area = np.linalg.norm(tr - tl) * np.linalg.norm(bl - tl)
            if current_area / ROI_area > OCR_THRESHOLD:
                text.append(detected_region[1])

    return text


if __name__ == "__main__":
    st.write("""
    # Car Plate Number Recognition App
    """)

    ### Load Input Image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Convert the file to a numpy image
        image = Image.open(uploaded_file).convert("RGB")
        image = np.asarray(image)

        _ = st.image(image, caption="Original Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            image_np_with_detections, detections = object_detector(image)
            ROIs = extract_ROIs(image_np_with_detections, detections)

        if ROIs is not None:
            _ = st.image(image_np_with_detections, caption="Detection Results", use_column_width=True)
            text = apply_OCR(ROIs)
            text = " ".join(text)
            st.write(f"""
            ## Car Plate Number: {text}
            """)
        else:
            st.write("""
            ## Cannot detect car plate number.
            """)

