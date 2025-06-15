import os
import glob
import csv

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

from object_detection.utils import config_util, label_map_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

import easyocr


# --- Configurations ---

IMAGE_DIR = os.path.join(os.getcwd(), "dataset", "images", "test")
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]
THRESHOLD = 0.75
ROI_AREA_RATIO_THRESHOLD = 0.01                 # This threshold ensures to eliminate noises in the ROI
BBOX_DIR = os.path.join(os.getcwd(), "dataset", "bbox", "test")
CSV_PATH = os.path.join(os.getcwd(), "output_batchsize_5_steps_13k_threshold_0.65_ROI_0.3.csv")

# --- End ---


def initialize_params():
    params_dict = {}

    params_dict["PATH"] = {
        "custom_model": os.path.join(".", "custom_model"),
        "images": os.path.join(".", "dataset", "images"),
        "annotation": os.path.join(".", "dataset", "annotation")
    }

    params_dict["FILE"] = {
        "custom_config": os.path.join(params_dict["PATH"]["custom_model"], "pipeline.config"),
        "custom_label_map": os.path.join(params_dict["PATH"]["annotation"], "label_map.pbtxt"),
        "custom_train_tf_record": os.path.join(params_dict["PATH"]["annotation"], "train.record"),
        "custom_test_tf_record": os.path.join(params_dict["PATH"]["annotation"], "test.record"),
        "TF_record_generator": os.path.join(".", "GenerateTFRecord", "generate_tfrecord.py")
    }

    return params_dict


def object_detector(params, image_path):
    config = config_util.get_configs_from_pipeline_file(params["FILE"]["custom_config"])
    detection_model = model_builder.build(model_config=config["model"], is_training=False)

    ckpt_name = latest_ckpt(params["PATH"]["custom_model"])
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(params["PATH"]["custom_model"], ckpt_name)).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(params["FILE"]["custom_label_map"])

    img = cv2.imread(image_path)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["num_detections"] = num_detections

    # detection_classes should be ints
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"],
        detections["detection_classes"] + label_id_offset,
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=THRESHOLD,
        agnostic_mode=False
    )

    return image_np_with_detections, detections


def latest_ckpt(path):
    ckpt_max = "-1"
    dir = os.listdir(path)

    for fname in dir:
        if os.path.isfile(os.path.join(path, fname)) and fname.startswith("ckpt"):
            ckpt = fname.split(".")[0]              # ckpt-1.data-000000-of-00001 --> ckpt-1
            ckpt_nr = int(ckpt.split("-")[-1])      # ckpt-1 --> 1
            if int(ckpt_nr) > int(ckpt_max):
                ckpt_max = ckpt_nr

    return "ckpt-{}".format(ckpt_max)


def extract_ROIs(image, detections):
    height, width, _ = image.shape
    ROIs = [box for box, score in zip(detections["detection_boxes"], detections["detection_scores"]) if
            score > THRESHOLD]
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
            if current_area / ROI_area > ROI_AREA_RATIO_THRESHOLD:
                text.append(detected_region[1])

    return text


def main():
    params = initialize_params()
    print("Reached here")
    return
    if not os.path.exists(BBOX_DIR):
        os.makedirs(BBOX_DIR)

    # Detect image files in folder
    images = []
    for pat in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(IMAGE_DIR, pat)))

    results = {}
    for image_path in images:
        image_np_with_detections, detections = object_detector(params, image_path)

        ROIs = extract_ROIs(image_np_with_detections, detections)

        if ROIs is not None:
            text = apply_OCR(ROIs)
        else:
            text = []

        results[image_path] = "\n".join(text)

        plt.figure()
        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        img_name = os.path.basename(image_path)
        bbox_name = img_name.split(".")[0] + "_bbox." + img_name.split(".")[1]
        bbox_path = os.path.join(BBOX_DIR, bbox_name)
        plt.savefig(bbox_path, dpi=300, bbox_inches='tight')

    # Export to csv
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        # Write header row (optional)
        writer.writerow(["Image path", "Car Plate Number"])

        for path, text in results.items():
            writer.writerow([path, text])


if __name__ == "__main__":
    main()
    print("test")
