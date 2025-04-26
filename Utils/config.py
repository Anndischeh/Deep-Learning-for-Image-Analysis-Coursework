import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
root = "/content/working" # Or local path
dataset_path = "/content/input" # Or local path
S = 7  # length of the grid
B = 2  # number of bounding boxes per cell
C = 5  # number of categories
image_resize = 124  #448  # Image resize
linear_size = 1024 # Number of features
grayscale = False # grayscale
ntrain = 77100  # Max training samples
nval = 3283  # Max validation samples
BATCH_SIZE = 128
EPOCHS = 35
LOAD_PATH = "/content/working/weights"  # Path to load model
SAVE_PATH = "/content/working/weights"  # Path to save model
savelossmap = True
in_channels = 1 if grayscale else 3
learning_rate = 1e-3
weight_decay = 5e-4
lambda_coord = 5
lambda_noobj = 0.5
iou_threshold = 0.5
confidence_threshold = 0.5
segments = 2 #gradient checkpointing
model_name = 'YOLOv1_3'
architecture = [
    (7, 64, 2, 3),
    "maxpool",
    (3, 192, 1, 1),
    "maxpool",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "maxpool",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "maxpool",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    [(3, 1024, 1, 1), 2],
]