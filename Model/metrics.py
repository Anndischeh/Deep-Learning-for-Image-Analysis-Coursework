import torch
from Utils.utils import IoU
from Utils.config import device, iou_threshold, confidence_threshold
import numpy as np
import warnings


class mAP_tool():

    def __init__(self, S, B, C, iou_threshold=iou_threshold, confidence_threshold=confidence_threshold):
        self.S, self.B, self.C = S, B, C
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

    def __call__(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1, self.S, self.S, self.C + self.B * 5)
        truth_boxes = y_true[..., self.C:self.C + 5]
        pred_boxes = y_pred[..., self.C:self.C + 5]
        truth_classes = torch.argmax(y_true[..., :self.C], dim=-1)
        pred_classes = torch.argmax(y_pred[..., :self.C], dim=-1)

        temp_ious = IoU(pred_boxes, truth_boxes, scale=self.S)
        for k in range(5, self.B * 5, 5):
            boxes = y_pred[..., self.C + k:self.C + 5 + k]
            iou = IoU(boxes, truth_boxes, scale=self.S)
            conditionnal = (iou > temp_ious).unsqueeze(-1).expand(-1, -1, -1, 5)
            temp_ious = torch.max(iou, temp_ious)
            pred_boxes = conditionnal * boxes + conditionnal.logical_not() * pred_boxes

        pred_ious = IoU(pred_boxes, truth_boxes, scale=self.S)
        confidences = pred_boxes[..., 4]
        pred_certain = (confidences >= self.confidence_threshold)

        positive = (pred_ious >= self.iou_threshold) * (truth_classes == pred_classes)
        TP = (pred_certain * positive).int()
        FP = (pred_certain * positive.logical_not()).int()
        object_present = y_true[..., self.C + 4]

        return torch.stack([object_present, confidences, TP, FP, truth_classes], dim=-1).reshape(-1, 5).cpu().numpy()
    

def calculate_mAP(AP):    
    mAP = (np.mean(AP)/0.005)
    return mAP

