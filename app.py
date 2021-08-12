import streamlit as st
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
from PIL import Image, ImageOps

import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2 
import numpy as np
from matplotlib import pyplot as plt

st.write("""
         # Nutrition Fact Table Extraction
         """
         )

file = st.file_uploader("Please upload a nutrition table image", type=["jpg", "png"])

#####
CUSTOM_MODEL_NAME = 'my_ssd_mobilnet' 
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

if file is None:
    st.text("Please upload an image file")
else: 

    IMAGE_PATH = file
    image = Image.open(file)
    #img = cv2.imread(IMAGE_PATH)
    #st.image(image, use_column_width=True)
    image_np = np.array(image)
    #st.text(image_np.shape)
    st.image(image, use_column_width=True)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    width = image_np.shape[1]
    height = image_np.shape[0]
    #Coordinates of detected objects
    ymin = int(detections['detection_boxes'][0][0]*height)
    xmin = int(detections['detection_boxes'][0][1]*width)
    ymax = int(detections['detection_boxes'][0][2]*height)
    xmax = int(detections['detection_boxes'][0][3]*width)
    crop_img = image_np[ymin:ymax, xmin:xmax]
    if detections['detection_scores'][0] < 0.5:
        crop_img.fill(0)
    
    st.image(crop_img, use_column_width=True)

#####

