import os
import shutil
import random
import subprocess
import glob
import csv
from tqdm import tqdm

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

from object_detection.utils import config_util, label_map_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

import easyocr


def split_images(image_path, train_ratio=0.8):
    images = os.listdir(os.path.join(image_path, "images"))
    bboxes = os.listdir(os.path.join(image_path, "annotations"))

    num_images = len(images)
    ind_list = range(num_images)
    random.seed(42)
    random.shuffle(ind_list)

    # Training dataset
    train_path = os.path.join(image_path, "train")
    os.makedirs(train_path, exist_ok=True)

    num_train_imgs = int(num_images * train_ratio)
    for ind in tqdm(ind_list[:num_train_imgs], desc="Copying images to training set"):
        img_src  = os.path.join(image_path, "images", images[ind])
        bbox_src = os.path.join(image_path, "annotations", bboxes[ind])
        shutil.move(img_src, train_path)
        shutil.move(bbox_src, train_path)

    # Testing dataset
    test_path = os.path.join(image_path, "test")
    os.makedirs(test_path, exist_ok=True)
    for ind in tqdm(ind_list[num_train_imgs:], desc="Copying images to test set"):
        img_src = os.path.join(image_path, "images", images[ind])
        bbox_src = os.path.join(image_path, "annotations", bboxes[ind])
        shutil.move(img_src, test_path)
        shutil.move(bbox_src, test_path)

    # Deleting the parent path
    shutil.rmtree(os.path.join(image_path, "images"))
    shutil.rmtree(os.path.join(image_path, "annotations"))


def initialize_params(args_image_path):
    params_dict = {}

    params_dict["PATH"] = {
        "custom_model": os.path.join(".", "custom_model"),
        # where label_map and TF records are saved. It is different from the one in original dataset.
        "annotation": os.path.join(".", "dataset", "annotation"),
        # Where images with detection boxes are saved.
        "BBOX_DIR": os.path.join(os.getcwd(), "dataset", "results", "bbox"),
        "IMAGE_TEST_DIR": args_image_path, #os.path.join(os.getcwd(), "dataset", "images", "test"),
        # A file saving car plate numbers
        "CSV_PATH": os.path.join(os.getcwd(), "dataset", "results", "car_plate_numbers.csv")
    }

    params_dict["FILE"] = {
        "custom_config": os.path.join(params_dict["PATH"]["custom_model"], "pipeline.config"),
        "custom_label_map": os.path.join(params_dict["PATH"]["annotation"], "label_map.pbtxt"),
        "custom_train_tf_record": os.path.join(params_dict["PATH"]["annotation"], "train.record"),
        "custom_test_tf_record": os.path.join(params_dict["PATH"]["annotation"], "test.record")
    }

    params_dict["CONSTANT"] = {
        "IMAGE_EXTENSIONS": ["*.jpg", "*.jpeg", "*.png"],
        "THRESHOLD": 0.75,
        "ROI_AREA_RATIO_THRESHOLD": 0.01,  # This threshold ensures to eliminate noises in the ROI
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
        min_score_thresh=params["CONSTANT"]["THRESHOLD"],
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


def extract_ROIs(image, detections, threshold=0.75):
    height, width, _ = image.shape
    ROIs = [box for box, score in zip(detections["detection_boxes"], detections["detection_scores"]) if
            score > threshold]
    if len(ROIs) == 0:
        return None
    result_ROIs = []
    for ROI in ROIs:
        absolute_ROI = ROI * [height, width, height, width]
        absolute_ROI = [int(coord) for coord in absolute_ROI]
        absolute_ROI = image[absolute_ROI[0]:absolute_ROI[2], absolute_ROI[1]:absolute_ROI[3], :]
        result_ROIs.append(absolute_ROI)

    return result_ROIs


def apply_OCR(ROIs, roi_area_ratio_threshold=0.01):
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
            if current_area / ROI_area > roi_area_ratio_threshold:
                text.append(detected_region[1])

    return text


def test(args_image_path):
    params = initialize_params(args_image_path)
    if not os.path.exists(params["PATH"]["BBOX_DIR"]):
        os.makedirs(params["PATH"]["BBOX_DIR"])

    # Detect image files in folder
    images = []
    for pat in params["CONSTANT"]["IMAGE_EXTENSIONS"]:
        images.extend(glob.glob(os.path.join(params["PATH"]["IMAGE_TEST_DIR"], pat)))

    results = {}
    for image_path in tqdm(images, desc="Testing"):
        image_np_with_detections, detections = object_detector(params, image_path)

        ROIs = extract_ROIs(image_np_with_detections, detections, threshold=params["CONSTANT"]["THRESHOLD"])

        if ROIs is not None:
            text = apply_OCR(ROIs, roi_area_ratio_threshold=params["CONSTANT"]["ROI_AREA_RATIO_THRESHOLD"])
        else:
            text = []

        img_name = os.path.basename(image_path)
        bbox_name = img_name.split(".")[0] + "_bbox." + img_name.split(".")[1]
        bbox_path = os.path.join(params["PATH"]["BBOX_DIR"], bbox_name)
        results[bbox_path] = "\n".join(text)

        plt.figure()
        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(bbox_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Export to csv
    with open(params["PATH"]["CSV_PATH"], "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        # Write header row (optional)
        writer.writerow(["Image path", "Car Plate Number"])

        for path, text in results.items():
            writer.writerow([path, text])


def main(args=None):

    ### TRAIN MODE
    if args.is_train:
        if ("images" not in os.listdir(args.image_path)) or ("annotations" not in os.listdir(args.image_path)):
            print("WARNING: Image directory does not include \"images\" or \"annotations\" folders.")
            print("         Splitting process cannot be executed.\n")
            print("Script terminated.\n")
            return

        # Split dataset into train and test sets
        split_images(args.image_path, train_ratio=args.train)

        # Run object_detector.py script
        cmd = [
            "python",
            "./Object_Detection/object_detector.py",
            "--is_train",
            "--img_classes", "licence"
            "--batch_size", args.batch_size,
            "--num_steps", args.num_steps
        ]
        subprocess.run(cmd, check=True)

    ### TEST MODE
    else:
        test(args.image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_train",
        help="Which phase to run [eg. train or inference]",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--image_path",
        help="Path to the dataset. This folder has 2 subfolders: images and annotations",
        default=r".\dataset",
        type=str
    )
    parser.add_argument(
        "--train",
        help="How much / Which ratio of the training images do you want to split?",
        default=0.8,
        type=float
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size to train model",
        default=5,
        type=int
    )
    parser.add_argument(
        "--num_steps",
        help="Number of steps to train model",
        default=13000,
        type=int
    )

    args = parser.parse_args()

    main(args)
    print("Finish!!")
