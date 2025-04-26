import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import pandas as pd
import json
import cv2
from Utils.config import device
from Utils.utils import complete_path
import matplotlib.pyplot as plt
from Utils.utils import IoU, get_grid, get_affine_item
from Utils.config import root


class COCODataset(Dataset):
    """
    Loads and preprocesses the COCO dataset for object detection.
    """

    def __init__(self, mode, dataset_path, S, B, categories, grayscale=False, max_nb_datapoints=None,
                 image_resize=448, in_channels=3):  # Added in_channels
        """
        Initializes the COCODataset object.

        Args:
            mode (str): 'train', 'val', or 'test'.
            dataset_path (str): Path to the COCO dataset.
            S (int): Grid size.
            B (int): Number of bounding boxes per cell.
            categories (Categories): Categories object.
            grayscale (bool, optional): Convert images to grayscale. Defaults to False.
            max_nb_datapoints (int, optional): Maximum number of data points to load. Defaults to None.
            image_resize (int, optional): Size to resize images. Defaults to 448.
            in_channels (int, optional): Number of input channels. Defaults to 3.  Added this!
        """
        assert mode in ['train', 'val', 'test'], "mode must be either 'train', 'val' or 'test'"

        self.images_path = complete_path(dataset_path)
        self.annotations_path = complete_path(dataset_path) + 'annotations/instances_' + mode + '2017.json'
        self.image_resize = image_resize
        self.mode = mode
        self.categories = categories
        self.grayscale = grayscale
        self.S = S
        self.B = B
        self.in_channels = in_channels  # Store in_channels
        self.C = len(self.categories) # this was moved here to make it available early.

        if self.mode == 'test':
            with open(complete_path(dataset_path) + 'annotations/image_info_test2017.json', 'r') as f:
                temp = json.load(f)
                self.images = pd.DataFrame(temp['images']).loc[:, ['id', 'file_name', 'width', 'height']].set_index('id')
                del temp
        else:
            with open(self.annotations_path, 'r') as f:
                temp = json.load(f)
                self.images = pd.DataFrame(temp['images']).loc[:, ['id', 'file_name', 'width', 'height']].set_index('id')
                self.annotations = pd.DataFrame(temp['annotations']).loc[:, ['image_id', 'bbox', 'category_id', 'iscrowd', 'area']]
                del temp

            self.annotations = self.annotations[self.annotations['category_id'].isin(self.categories.df['id'])]
            self.annotations = self.annotations[self.annotations['area'] > 1024]
            self.annotations = self.annotations[self.annotations['iscrowd'] == 0]
            index = pd.MultiIndex.from_frame(self.annotations.reset_index().loc[:, ['image_id', 'index']])
            self.annotations = self.annotations.set_index(index).loc[:, ['bbox', 'category_id']]
            self.annotations['category_id'] = self.annotations['category_id'].map(self.categories.df.reset_index().set_index('id')['index'])
            self.images = self.images[self.images.index.isin(pd.unique(self.annotations.index.get_level_values(0)))]

        nb_images = len(self.images)
        self.N = nb_images if not max_nb_datapoints else min(nb_images, max_nb_datapoints)

        self.images = self.images.iloc[:self.N, :]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images.iloc[idx]
        image_content = io.imread(f"{self.images_path}{self.mode}2017/{image['file_name']}")

        if self.grayscale:
            if image_content.ndim == 3:
                image_content = rgb2gray(image_content)
            image_content = torch.from_numpy(image_content).to(device).unsqueeze(0).double()
        else:
            if image_content.ndim == 2:
                image_content = np.stack((image_content,) * 3, axis=-1)
            image_content = torch.from_numpy(image_content).to(device).permute(2, 0, 1).double()

        image_width = image['width']
        image_height = image['height']

        operations = [
            transforms.Resize([self.image_resize, self.image_resize]),
            lambda x: x if self.grayscale else x / 255,
        ]

        is_hflipped = False
        affine_params = (0., (0, 0), 0., (0., 0.))

        if self.mode == 'train':
            operations.append(transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.4, hue=0.1))
            if np.random.random() > 0.5:
                operations.append(transforms.functional.hflip)
                is_hflipped = True
            if np.random.random() > 0.5:
                affine_params = transforms.RandomAffine.get_params(degrees=[-30., 30.], translate=[0.1, 0.1],
                                                                     scale_ranges=[0.95, 1.25], shears=[0., 0.],
                                                                     img_size=(self.image_resize, self.image_resize))
                operations.append(lambda img: transforms.functional.affine(img, *affine_params))

        transform = transforms.Compose(operations)
        image_content = transform(image_content)

        if self.mode != 'test':
            annot = self.annotations.loc[image.name, :]
            label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B), dtype=torch.float64)

            iou_matrix = np.zeros((self.S, self.S))

            for [x, y, w, h], category_id in annot.values:
                x = x / image_width
                y = y / image_height
                w = w / image_width
                h = h / image_height

                def coord_conv(x, y, w, h):
                    if is_hflipped:
                        x, y, w, h = 1 - x - w, y, w, h

                    if affine_params != (0., (0, 0), 0., (0., 0.)):
                        x1, x2, x3, x4 = x, x + w, x, x + w
                        y1, y2, y3, y4 = y, y, y + h, y + h
                        angle, (tx, ty), scale, _ = affine_params
                        angle_deg = angle * np.pi / 180.
                        theta = torch.tensor([[np.cos(angle_deg), -1.0 * np.sin(angle_deg), 0.],
                                               [np.sin(angle_deg), np.cos(angle_deg), 0.], [0., 0., 1.]],
                                              dtype=torch.float64)
                        T = lambda dx, dy: torch.tensor([[1., 0., dx], [0., 1., dy], [0., 0., 1.]], dtype=torch.float64)

                        def convert_point(x, y):
                            XY = torch.tensor([x, y, 1.], dtype=torch.float64)
                            scaling = torch.eye(3, dtype=torch.float64) * scale
                            scaling[-1, -1] = 1.
                            for mat in [T(-1 / 2, -1 / 2), scaling, theta, T(1 / 2, 1 / 2),
                                        T(tx / image_width, ty / image_height)]:
                                XY = torch.matmul(mat, XY)
                                x, y, _ = XY
                            return x, y

                        x1, y1 = convert_point(x1, y1)
                        x2, y2 = convert_point(x2, y2)
                        x3, y3 = convert_point(x3, y3)
                        x4, y4 = convert_point(x4, y4)
                        xmax, xmin = max(x1, x2, x3, x4), min(x1, x2, x3, x4)
                        ymax, ymin = max(y1, y2, y3, y4), min(y1, y2, y3, y4)
                        return xmin, ymin, xmax - xmin, ymax - ymin
                    return x, y, w, h

                x, y, w, h = coord_conv(x, y, w, h)
                if x > 1 or y > 1 or (x + w) < 0 or (y + h) < 0:
                    continue
                x, y = min(max(x, 0), 1), min(max(y, 0), 1)
                w, h = min(w, 1 - x), min(h, 1 - y)
                x = x + w / 2
                y = y + h / 2

                i = int(y * self.S)
                j = int(x * self.S)

                iou = IoU([x * self.S, y * self.S, w * self.S, h * self.S], [j + 1 / 2, i + 1 / 2, 1, 1])
                if iou > iou_matrix[i, j]:
                    iou_matrix[i, j] = iou
                    x_cell = x * self.S - j
                    y_cell = y * self.S - i
                    label_matrix[i, j, :self.C] = torch.zeros(self.C, dtype=torch.float64)
                    label_matrix[i, j, category_id] = 1.
                    label_matrix[i, j, self.C:self.C + 5] = torch.as_tensor([x_cell, y_cell, w, h, 1],
                                                                              dtype=torch.float64)

            sample = {'image': image_content, 'label': label_matrix, 'id': image.name, 'is_hflipped': is_hflipped,
                      'affine_params': affine_params}
        else:
            sample = {'image': image_content, 'id': image.name, 'is_hflipped': is_hflipped,
                      'affine_params': affine_params}

        return sample

class Sample:
    """
    Represents a single sample from the dataset for visualization.
    """

    def __init__(self, sample, dataset, compute_label=False, model=None, image_width=None, image_height=None):
        """
        Initializes the Sample object.

        Args:
            sample (dict): {'image': Tensor, 'label': Tensor, 'id': int, 'is_hflipped': bool, 'affine_params': tuple}
            dataset (COCODataset): COCODataset object.
            compute_label (bool, optional): Whether to compute the predicted label using a model. Defaults to False.
            model (nn.Module, optional): YOLOv1 model for prediction. Defaults to None.
            image_width (int, optional): The width of the image. If None, defaults to the width from dataset.images.
            image_height (int, optional): The height of the image. If None, defaults to the height from dataset.images.
        """
        self.grayscale = dataset.grayscale
        self.image = sample['image']
        id = sample['id']
        if torch.is_tensor(id):
            id = id.tolist()

        self.im_path = f"{dataset.images_path}{dataset.mode}2017/"

        # Use provided width and height, otherwise lookup from dataset.images
        if image_width is not None and image_height is not None:
            self.im_width = image_width
            self.im_height = image_height
            self.im_name = "000000000139.jpg" #dummy image name
        else:
            self.im_name = dataset.images.loc[id, 'file_name']
            self.im_width, self.im_height = dataset.images.loc[id, ['width', 'height']].values

        operations = []
        if sample['is_hflipped']:
            operations.append(transforms.functional.hflip)
        if sample['affine_params'] != (0., (0, 0), 0., (0., 0.)):
            angle, (tx, ty), scale, _ = sample['affine_params']
            tx, ty = tx * self.im_width / dataset.image_resize, ty * self.im_height / dataset.image_resize
            operations.append(lambda img: transforms.functional.affine(img, angle, (tx, ty), scale, 0.))
        self.augment = transforms.Compose(operations)
        self.angle = sample['affine_params'][0]
        self.im_resize = dataset.image_resize
        self.S, self.B, self.C = dataset.S, dataset.B, dataset.C
        self.categories = dataset.categories

        if 'label' in sample:
            self.label = sample['label'].cpu().numpy()
        else:
            self.label = None
        if compute_label:
            self.pred_label = model(self.image.unsqueeze(0).to(device)).reshape(self.S, self.S,
                                                                                self.C + self.B * 5).cpu().detach().numpy()
            self.max_bbox_ind = np.argmax(self.pred_label[:, :, self.C + 4::5], axis=2)
        else:
            self.pred_label = None

        if self.grayscale:
            self.image = self.image.cpu().repeat(3, 1, 1).permute(1, 2, 0).numpy()
        else:
            self.image = self.image.cpu().permute(1, 2, 0).numpy()

    def show(self, rescaled=True, conf_threshold=0.5, figsize=(10, 10), ax=None, lbl_txt_scale=0.5, savefig=True):
        """
        Displays the image with bounding boxes.

        Args:
            rescaled (bool, optional): Whether to rescale the image to its original size. Defaults to True.
            conf_threshold (float, optional): Confidence threshold for displaying bounding boxes. Defaults to 0.5.
            figsize (tuple, optional): Figure size. Defaults to (10, 10).
            ax (matplotlib.axes._axes.Axes, optional): Matplotlib Axes object. Defaults to None.
            lbl_txt_scale (float, optional): Scaling factor for labels. Defaults to 0.5.
        """
        if not ax:
            plt.figure(figsize=figsize)
        if rescaled:
            if self.grayscale:
                image_content = io.imread(f"{self.im_path}{self.im_name}", as_gray=True) * 255
                image_content = np.stack((image_content,) * 3, axis=0)
            else:
                image_content = io.imread(f"{self.im_path}{self.im_name}")
            if image_content.ndim == 2:
                image_content = np.stack((image_content,) * 3, axis=0)
            else:
                image_content = image_content.transpose(2, 0, 1)
            display_image = self.augment(torch.Tensor(image_content)).permute(1, 2, 0).numpy()
        else:
            display_image = self.image.copy() * 255
        scale = lbl_txt_scale * (np.sqrt(np.prod(display_image.shape[:2])) / 600)
        height, width = display_image.shape[:2]

        k = [None] * 2
        for is_pred, label in enumerate([self.label, self.pred_label]):
            if not label is None:
                k[is_pred] = 0
                for i in range(self.S):
                    for j in range(self.S):
                        temp = label[i, j, :self.C].argmax()

                        ind_bbox = 0 if not is_pred else self.max_bbox_ind[i, j]
                        confidence = label[i, j, self.C + ind_bbox + 4]
                        category_id = None if (confidence < conf_threshold) else temp

                        if not category_id is None:
                            k[is_pred] += 1
                            category_name = self.categories.category_dict[category_id]
                            x_cell, y_cell, w, h = label[i, j, self.C + ind_bbox:self.C + ind_bbox + 4]

                            x, y = (x_cell + j) * width / self.S, (y_cell + i) * height / self.S
                            w, h = w * width, h * height
                            x, y = x - w / 2, y - h / 2

                            if not is_pred:
                                temp = display_image.copy()
                                cv2.rectangle(temp, (int(x), int(y)), (int(x + w), int(y + h)),
                                              self.categories.category_colors[category_id], -1)
                                mask = np.zeros(temp.shape)
                                cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), (1, 1, 1), -1)
                                display_image = cv2.addWeighted(display_image, 0.65, temp, 0.35, 0) * mask + display_image * (
                                            1 - mask)
                            cv2.rectangle(display_image, (int(x), int(y)), (int(x + w), int(y + h)),
                                          self.categories.category_colors[category_id], 2)

                            text = category_name + f"({round(confidence, 3)})"
                            (wt, ht), _ = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                            cv2.rectangle(display_image, (int(x), int(y + ht)), (int(x + wt), int(y)),
                                          self.categories.category_colors[category_id], -1)
                            cv2.putText(display_image, text, (int(x), int(y + ht - 2)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                                        (0, 0, 0), 1)

        im_dim_desc = f"{width}x{height}" if rescaled else ""
        was_pred = k[1] is not None;
        was_target = k[0] is not None
        pred_target_desc = (f" {k[0]} present" if was_target else "") + ("," if (was_pred and was_target) else "") + (
            f" {k[1]} predicted" if was_pred else "")
        if not ax:
            plt.imshow(display_image / 255)
            plt.xlabel(f"{self.im_name} {im_dim_desc}{pred_target_desc}", fontsize=11)
            plt.xticks([])
            plt.yticks([])
            if savefig:
                plt.savefig(f"{root}/plots/sample_images.png")
            plt.show()
        else:
            ax.imshow(display_image / 255)
            ax.set_xlabel(f"{im_dim_desc}{pred_target_desc}")
            ax.set_xticks([])
            ax.set_yticks([])


def show_batch(sample_batched, dataset, savefig=True, rescaled=False, compute_label=False, figsize=(10, 8), lbl_txt_scale=0.5, model=None):
    """
    Displays a batch of images with bounding boxes.

    Args:
        sample_batched (dict): Batch of samples from the DataLoader.
        dataset (COCODataset): COCODataset object.
        rescaled (bool, optional): Whether to rescale the images. Defaults to False.
        compute_label (bool, optional): Whether to compute predicted labels. Defaults to False.
        figsize (tuple, optional): Figure size. Defaults to (10, 8).
        lbl_txt_scale (float, optional): Label text scale. Defaults to 0.5.
        model (nn.Module, optional): YOLOv1 model for prediction. Defaults to None.
    """
    plt.figure(figsize=figsize)
    batch_size = len(sample_batched['image'])
    plt.rc('axes', labelsize=int(-batch_size / 10 + 12))
    plt.suptitle(f"Batch of {batch_size} {dataset.mode} images")
    nrows, ncols = get_grid(batch_size)
    for i in range(batch_size):
        keys = ['image']
        if 'label' in sample_batched:
            keys.append('label')
        keys.extend(['id', 'is_hflipped'])
        d = {key: sample_batched[key][i] for key in keys}
        d['affine_params'] = get_affine_item(i, sample_batched['affine_params'])
        sample = Sample(d, dataset, compute_label=compute_label, model=model)
        ax = plt.subplot(nrows, ncols, i + 1)
        sample.show(rescaled=rescaled, ax=ax, lbl_txt_scale=lbl_txt_scale)
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{root}/plots/batch_sample_images.png")
    plt.show()