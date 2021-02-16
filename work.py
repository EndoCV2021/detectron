import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        print (evaluator_type)
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        dist = MetadataCatalog.get(dataset_name).distributed
        if evaluator_type in ['coco', 'coco_panoptic_seg']:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, dist, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, dist, output_dir=output_folder)
        print(evaluator_list)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
from pycocotools import mask
import cv2


def get_polyp_dicts(data_dirs):
    '''
        Dataset 만드는 부분
        input:
            data_dirs: List
        return:
            dataset_dicts: Dictionary
    '''
    base_dir = './trainData_EndoCV2021_5_Feb2021/'
    dataset_dicts = []

    cnt = 0
    for data_dir in data_dirs:
        print ("LOAD " + data_dir + " STARTED.........")

        data_dir_full_path = os.path.join (base_dir, data_dir)
        dir_names = sorted (os.listdir (data_dir_full_path))

        bbox_dir = os.path.join (data_dir_full_path, dir_names [0])
        bbox_img_dir = os.path.join (data_dir_full_path, dir_names [1])
        image_dir = os.path.join (data_dir_full_path, dir_names [2])
        mask_dir = os.path.join (data_dir_full_path, dir_names [3])

        image_filenames = sorted (os.listdir (image_dir))

        for image_filename in image_filenames:
            record = {}

            image_filename_full_path = os.path.join (image_dir, image_filename)
            height, width = cv2.imread (image_filename_full_path).shape [:2]

            record ["file_name"] = image_filename_full_path
            record ["height"] = height
            record ["width"] = width
            record ["image_id"] = cnt

            cnt += 1

            fn = os.path.splitext (image_filename) [0]
            bbox_filename_full_path = os.path.join (bbox_dir, fn + "_mask.txt")
            mask_filename_full_path = os.path.join (mask_dir, fn + "_mask.jpg")

            objs = []

            _mask = cv2.imread (mask_filename_full_path)
            _mask = cv2.cvtColor (_mask, cv2.COLOR_BGR2GRAY)

            # _mask를 출력해보면 binary가 아님.
            # 0, 1, 2, ... , 8 그리고 247, 248, ... , 255 값이 들어있는 것으로 확인.
            # binarization이 필요.
            _mask [_mask < 128] = 0
            _mask [_mask > 128] = 1

            with open (bbox_filename_full_path) as f:
                contents = f.readlines ()

                for anno in contents:
                    anno = anno.replace ("\n", "")
                    strings = anno.split (' ')
                    # annotation 형태: polyp x_min y_min x_max y_max

                    if strings [0] == 'polyp':
                        # polyp이 있는 경우만 데이터셋에 추가
                        x_min = int (strings [1])
                        y_min = int (strings [2])
                        x_max = int (strings [3])
                        y_max = int (strings [4])

                        # bounding box와 segmenation mask이 둘 다 1인 부분 찾기
                        # 하나의 image에 여러 polyp이 있는 경우
                        _bbox_img = np.zeros ((height, width))
                        _bbox_img [y_min:y_max, x_min:x_max] = 1

                        _mask_bbox = _bbox_img * _mask

                        # binary segmentation mask를 detectron2에서 요구하는 형식(COCO’s compressed RLE format) 으로 변환
                        _mask_bbox = _mask_bbox.astype ('uint8')
                        _mask_dict = mask.encode (np.asarray (_mask_bbox, order="F"))

                        obj = {
                            "bbox": [x_min, y_min, x_max, y_max],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": 0,
                            "iscrowd": 0,
                            "segmentation": _mask_dict
                        }
                        objs.append (obj)

            record ["annotations"] = objs
            dataset_dicts.append (record)
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog

def setup(args):
    """
    Create configs and perform basic setups.
    """
    train_data_dirs = ['data_C2', 'data_C3', 'data_C4', 'data_C5']
    val_data_dirs = ['data_C1']

    DatasetCatalog.register ("polyp_train", lambda: get_polyp_dicts (train_data_dirs))
    MetadataCatalog.get ("polyp_train").set (thing_classes=["polyp"])

    DatasetCatalog.register ("polyp_val", lambda: get_polyp_dicts (val_data_dirs))
    MetadataCatalog.get ("polyp_val").set (thing_classes=["polyp"], evaluator_type=("coco"), distributed=True)

    # polyp_metadata = MetadataCatalog.get ("polyp_train")
    #
    # dataset_dicts = get_polyp_dicts (train_data_dirs)

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.TRAIN = ("polyp_train",)
    cfg.DATASETS.TEST = ("polyp_val",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    # cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_50_FPN_3x/137849486/model_final_4cafe0.pkl"  # initialize from model zoo
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    # print(cfg.MODEL)
    os.makedirs (cfg.OUTPUT_DIR, exist_ok=True)
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000    # 300 iterations 정도면 충분합니다. 더 오랜 시간도 시도해보세요.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # 풍선 데이터셋과 같이 작은 데이터셋에서는 이정도면 적당합니다.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # a = MetadataCatalog.get('polyp_val')
    # print(a.evaluator_type)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()