import os
import numpy as np
import sys

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")


# import PointRend project
from detectron2.projects import point_rend



def pointrend(im):
    """
    PointRend on single image
    """
    cfg = get_cfg()
    # Add PointRend-specific config
    point_rend.add_pointrend_config(cfg)
    # Load a config from file
    cfg.merge_from_file("projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    return point_rend_result


def pr_dir(src_dir, tgt_dir):
    """
    PointRend on dir
    """
    img_name_list = os.listdir(src_dir)
    print(f"Running PR for {src_dir}, there are {len(img_name_list)} images in total")
    for img_name in img_name_list:
        img_file = os.path.join(src_dir, img_name)
        img = cv2.imread(img_file)
        img_file_o = os.path.join(tgt_dir, img_name)
        pr_img = pointrend(img)
        cv2.imwrite(img_file_o, pr_img)
    print(f"Finished for {src_dir}")


### The following code is to run pointrend and generate 8 channel results are Tien asked

def inference(img):
    cfg = get_cfg()
    # Add PointRend-specific config
    point_rend.add_pointrend_config(cfg)
    # Load a config from file
    cfg.merge_from_file("projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)
    return outputs


def pr_8c(img):
    outputs = inference(img)
    return outputs


def pr_8c_dir(src_dir, tgt_dir):
    img_name_list = os.listdir(src_dir)
    print(f"Running PR for {src_dir}, there are {len(img_name_list)} images in total")
    for img_name in img_name_list:
        img_file = os.path.join(src_dir, img_name)
        img = cv2.imread(img_file)
        img_file_o = os.path.join(tgt_dir, img_name)
        pr_img = pr_8c(img)
        cv2.imwrite(img_file_o, pr_img)
    print(f"Finished for {src_dir}")


if __name__ == "__main__":
    src_dir = sys.argv[1]
    tgt_dir = sys.argv[2]
    pr_8c_dir(src_dir=src_dir, tgt_dir=tgt_dir)