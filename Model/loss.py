import torch
import torch.nn as nn
from Utils.utils import IoU
from Utils.config import device


class YOLOv1_loss(nn.Module):
    """
    YOLOv1 loss function.
    """

    def __init__(self, S, B, C, lambda_coord, lambda_noobj, weights, eps=1e-8):
        """
        Initializes the YOLOv1_loss object.

        Args:
            S (int): Grid size.
            B (int): Number of bounding boxes per cell.
            C (int): Number of categories.
            lambda_coord (float): Weight for bounding box coordinate loss.
            lambda_noobj (float): Weight for no object confidence loss.
            weights (np.ndarray): Weights for class probabilities.
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.
        """
        super(YOLOv1_loss, self).__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.eps = eps
        self.weights = torch.from_numpy(weights).double().to(device)

    def __repr__(self):
        return f"<main.YOLOv1_loss object: S={self.S}, B={self.B}, C={self.C}, lambda_coord={self.lambda_coord}, lambda_noobj={self.lambda_noobj}, eps={self.eps}>"

    def forward(self, y_pred, y_true):
        """
        Calculates the YOLOv1 loss.

        Args:
            y_pred (torch.Tensor): Predicted output from the model (batch_size, S*S*(C+B*5)).
            y_true (torch.Tensor): Ground truth labels (batch_size, S, S, C+B*5).

        Returns:
            torch.Tensor: Mean loss over the batch.
        """
        # y_pred: (batch_size, SS(C+B5))
        # y_true: (batch_size, S, S, C+B5)
        y_pred = y_pred.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # We follow the loss formula given in the paper

        # Selecting the highest IoU among the B predictors in each cell
        # truth_boxes, pred_boxes: (batch_size, S, S, 4)
        truth_boxes = y_true[..., self.C:self.C + 5]
        pred_boxes = y_pred[..., self.C:self.C + 5]

        # bboxes in [x_cell, y_cell, w, h] format, got to change x and y to be relative to the whole image for calculating IoU
        temp_ious = IoU(pred_boxes, truth_boxes, scale=self.S)
        for k in range(5, self.B * 5, 5):
            boxes = y_pred[..., self.C + k:self.C + 5 + k]
            iou = IoU(boxes, truth_boxes, scale=self.S)
            conditionnal = (iou > temp_ious).unsqueeze(-1).expand(-1, -1, -1, 5)
            temp_ious = torch.max(iou, temp_ious)
            pred_boxes = conditionnal * boxes + conditionnal.logical_not() * pred_boxes

        # Applying the 1i^obj function as described in the paper, which becomes 1ij^obj because pred_boxes is the matrix of the best boxes for each cell
        object_present = y_true[..., self.C + 4]
        onei = object_present.unsqueeze(-1).expand(-1, -1, -1, 5)  # expand is very memory efficient

        confidence_noobj_pred = pred_boxes[..., 4] * (1 - object_present)
        confidence_noobj = truth_boxes[..., 4] * (1 - object_present)
        pred_boxes *= onei

        bboxes_loss = (
                (pred_boxes[..., 0] - truth_boxes[..., 0]) ** 2
                + (pred_boxes[..., 1] - truth_boxes[..., 1]) ** 2
                + (torch.sign(pred_boxes[..., 2]) * torch.sqrt(torch.abs(pred_boxes[..., 2]) + self.eps) - torch.sqrt(
            truth_boxes[..., 2] + self.eps)) ** 2
                + (torch.sign(pred_boxes[..., 3]) * torch.sqrt(torch.abs(pred_boxes[..., 3]) + self.eps) - torch.sqrt(
            truth_boxes[..., 3] + self.eps)) ** 2
        )

        confidences_obj_loss = (pred_boxes[..., 4] - truth_boxes[..., 4]) ** 2
        confidences_noobj_loss = (confidence_noobj_pred - confidence_noobj) ** 2

        classes_loss = torch.sum(
            self.weights.expand(truth_boxes.size(0), self.S, self.S, self.C) * object_present.unsqueeze(-1).expand(
                -1, -1, -1, self.C) * (y_pred[..., :self.C] - y_true[..., :self.C]) ** 2, -1)

        loss = (
                bboxes_loss * self.lambda_coord
                + confidences_obj_loss
                + confidences_noobj_loss * self.lambda_noobj
                + classes_loss
        )
        return torch.mean(torch.sum(loss, (2, 1)))  # Returns mean loss over batch