import argparse
import os
import shutil
import random
import time
import warnings
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from enum import Enum

# Apply the guided filter to the grayscale images as mentioned in the paper


def guided_filter(image, radius=1, epsilon=1e-6):
    """
    Apply a guided filter to the input image.

    Args:
        image: Grayscale image (2D numpy array).
        radius: Radius of the local window (default: 1 for 3x3 neighborhood).
        epsilon: Regularization parameter to stabilize `a_k` (default: 1e-6).

    Returns:
        Grayscale Filtered image (guided filtered version).
    """
    # Convert to float for calculations
    image = image.astype(np.float32) / 255.0
    h, w = image.shape  # height and width of the image

    # Create the ideal binary image using Otsu's threshold
    _, I_ideal = cv2.threshold(
        (image * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    I_ideal = I_ideal.astype(np.float32) / 255.0

    # Define kernel for filtering
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float32)

    # Compute mean and variance in the local window
    mean_I = (
        cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT) / kernel.sum()
    )
    mean_I_ideal = (
        cv2.filter2D(I_ideal, -1, kernel, borderType=cv2.BORDER_REFLECT) / kernel.sum()
    )

    mean_I_ideal_squared = (
        cv2.filter2D(I_ideal * I_ideal, -1, kernel, borderType=cv2.BORDER_REFLECT)
        / kernel.sum()
    )
    var_I_ideal = mean_I_ideal_squared - mean_I_ideal**2

    cov_I_Ideal = (
        cv2.filter2D(image * I_ideal, -1, kernel, borderType=cv2.BORDER_REFLECT)
        / kernel.sum()
        - mean_I * mean_I_ideal
    )

    # Compute coefficients a_k and b_k
    a_k = cov_I_Ideal / (var_I_ideal + epsilon)
    b_k = mean_I - a_k * mean_I_ideal

    # Compute mean of a_k and b_k
    mean_a_k = np.zeros((h, w), dtype=np.float32)
    mean_b_k = np.zeros((h, w), dtype=np.float32)

    # this approach does not handle edge cases well
    # mean_a_k = cv2.filter2D(a_k, -1, kernel, borderType=cv2.BORDER_REFLECT) / kernel.sum()
    # mean_b_k = cv2.filter2D(b_k, -1, kernel, borderType=cv2.BORDER_REFLECT) / kernel.sum()

    # this approach handles edge cases better
    num_elements_w = [3] * w
    num_elements_w[0] = 1
    num_elements_w[1] = 2
    num_elements_w[w - 2] = 2
    num_elements_w[w - 1] = 1

    num_elements_h = [3] * h
    num_elements_h[0] = 1
    num_elements_h[1] = 2
    num_elements_h[h - 2] = 2
    num_elements_h[h - 1] = 1

    for i in range(h):
        for j in range(w):
            sum_a_k = 0.0  # Reset sum_a_k for each (i, j) computation
            sum_b_k = 0.0  # Reset sum_b_k for each (i, j) computation

            for _i in range(i - 2, i + 1):  # Iterate over _i range
                if 0 <= _i <= h - 3:  # Ensure _i is within valid bounds
                    for _j in range(j - 2, j + 1):  # Iterate over _j range
                        if 0 <= _j <= w - 3:  # Ensure _j is within valid bounds
                            sum_a_k += a_k[_i + 1][_j + 1]
                            sum_b_k += b_k[_i + 1][_j + 1]

            mean_a_k[i][j] = sum_a_k / (num_elements_h[i] * num_elements_w[j])
            mean_b_k[i][j] = sum_b_k / (num_elements_h[i] * num_elements_w[j])

    # Compute final guided filtered image
    I_GF = mean_a_k * I_ideal + mean_b_k

    # Scale back to 8-bit range for saving
    return (I_GF * 255).astype(np.uint8)
    # return (I_ideal * 255).astype(np.uint8)


# This implementation is supposed to take more time to load the data. datasize = 2 X total_num_of_patches
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        """
        Custom dataset for loading patches and returning labels based on folder names.

        Args:
            root (str): Path to the root folder containing patches.
            transform (callable, optional): Transformations to be applied to the patches.
        """
        self.root = root
        self.transform = transform
        self.data = []

        # Create a mapping of folder names to labels
        self.class_to_idx = {
            folder: idx for idx, folder in enumerate(sorted(os.listdir(root)))
        }

        # Collect all patches and their labels
        for folder in os.listdir(root):
            folder_path = os.path.join(root, folder)
            if os.path.isdir(folder_path):
                original_path = os.path.join(folder_path, "original")
                guided_path = os.path.join(folder_path, "guided")

                if os.path.exists(original_path) and os.path.exists(guided_path):
                    for patch_name in os.listdir(original_path):
                        original_patch = os.path.join(original_path, patch_name)
                        guided_patch = os.path.join(guided_path, patch_name)

                        if os.path.exists(guided_patch):
                            label = self.class_to_idx[
                                folder
                            ]  # Infer label from folder name
                            self.data.append((original_patch, guided_patch, label))

    def __getitem__(self, index):
        """
        Retrieve an original-guided patch pair, apply transformations, and return the label.
        """
        original_path, guided_path, label = self.data[index]

        # Load the images
        original_patch = Image.open(original_path).convert("L")
        guided_patch = Image.open(guided_path).convert("L")
        # print("original_patch shape using Image: ", original_patch.size)

        to_tensor = transforms.ToTensor()
        original_patch = to_tensor(original_patch)  # shape: (1, H, W)
        guided_patch = to_tensor(guided_patch)  # shape: (1, H, W)
        original_patch = original_patch.squeeze(0)  # shape: (H, W)
        guided_patch = guided_patch.squeeze(0)  # shape: (H, W)
        # print("original_patch shape after tensor: ", original_patch.shape)
        # Stack the original and guided patches as two channels
        stacked_patch = torch.stack([original_patch, guided_patch], dim=0)

        # Apply transformations if defined
        if self.transform:
            stacked_patch = self.transform(stacked_patch)

        # print("stacked_patch shape: ", stacked_patch.shape)

        return stacked_patch, label

    def __len__(self):
        """
        Total number of patch pairs.
        """
        return len(self.data)


# This implementation is supposed to take less time to load the data. datasize = total_num_of_patches
# Preprocessing of original patch on the fly
class PatchDataset_2(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        """
        Custom dataset for loading patches and returning labels based on folder names.

        Args:
            root (str): Path to the root folder containing patches.
            transform (callable, optional): Transformations to be applied to the patches.
        """
        self.root = root
        self.transform = transform
        self.data = []

        # Create a mapping of folder names to labels
        self.class_to_idx = {
            folder: idx for idx, folder in enumerate(sorted(os.listdir(root)))
        }

        # Collect all patches and their labels
        for folder in os.listdir(root):
            folder_path = os.path.join(root, folder)
            if os.path.isdir(folder_path):
                original_path = os.path.join(
                    folder_path, "original"
                )  # original patch pathes
                # guided_path = os.path.join(folder_path, "guided")

                # if os.path.exists(original_path) and os.path.exists(guided_path):
                for patch_name in os.listdir(original_path):
                    original_patch = os.path.join(original_path, patch_name)
                    # guided_patch = os.path.join(guided_path, patch_name)

                    label = self.class_to_idx[folder]  # Infer label from folder name
                    self.data.append((original_patch, label))

    def __getitem__(self, index):
        """
        Retrieve an original patch, apply guided filter, apply training transformations, and return the (staked_ptach,label).
        """
        original_path, label = self.data[index]

        # Load the images
        original_patch = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        guided_patch = guided_filter(
            original_patch
        )  # Extract guided patch using guided filter

        # guided_patch = Image.open(guided_path).convert("L")
        # print("original_patch shape using Image: ", original_patch.size)

        to_tensor = transforms.ToTensor()
        original_patch = to_tensor(original_patch)  # shape: (1, H, W)
        guided_patch = to_tensor(guided_patch)  # shape: (1, H, W)
        original_patch = original_patch.squeeze(0)  # shape: (H, W)
        guided_patch = guided_patch.squeeze(0)  # shape: (H, W)
        # print("original_patch shape after tensor: ", original_patch.shape)
        # Stack the original and guided patches as two channels
        stacked_patch = torch.stack([original_patch, guided_patch], dim=0)

        # Apply transformations if defined
        if self.transform:
            stacked_patch = self.transform(stacked_patch)

        # print("stacked_patch shape: ", stacked_patch.shape)

        return stacked_patch, label

    def __len__(self):
        """
        Total number of patch pairs.
        """
        return len(self.data)


parser = argparse.ArgumentParser(description="Khanna et al, 2019 Training script")
parser.add_argument(
    "data",
    metavar="DIR",
    nargs="?",
    default="/Users/hemantpanchariya/Desktop/IIT_RPR/khanna_implementation/data",
    help="path to dataset (default: imagenet)",
)

parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    # default=256,
    default=100,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=5e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 5e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=1,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)

parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    main_worker(args.gpu, args, ngpus_per_node=1)


def main_worker(gpu, args, ngpus_per_node=1):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Using GPU: {} for training".format(args.gpu))

    # create model
    class model(nn.Module):
        def __init__(self, num_classes):
            super(model, self).__init__()
            self.conv1 = nn.Conv2d(2, 50, kernel_size=3, stride=1)
            self.bn1 = nn.BatchNorm2d(50)
            self.conv2 = nn.Conv2d(50, 50, kernel_size=3, stride=1)
            self.bn2 = nn.BatchNorm2d(50)
            self.conv3 = nn.Conv2d(50, 50, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(50)
            self.pool = nn.MaxPool2d(2, 2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(6 * 6 * 50, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            # x = torch.softmax(x, dim=1)
            return x

    num_classes = (
        7  # TO DO : MAKE IT DYNAMIC BASED ON THE NUMBER OF FOLDERS IN THE DATASET
    )
    print("Creating model")
    model = model(num_classes)
    print("Model created")

    # if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    #     print("using CPU, this will be slow")
    # elif args.gpu is not None and torch.cuda.is_available():
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)

    #     num_gpus = torch.cuda.device_count()
    #     print(f"Number of GPUs available: {num_gpus}")
    #     for i in range(num_gpus):
    #         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     model = model.to(device)
    # else:
    #     # DataParallel will divide and allocate batch_size to all available GPUs
    #     model = torch.nn.DataParallel(model).cuda()

    # if torch.cuda.is_available():
    #     if args.gpu:
    #         device = torch.device("cuda:{}".format(args.gpu))
    #         print(
    #             f"Device is moved to GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}"
    #         )
    #     else:
    #         device = torch.device("cuda")
    #         print("Device set to GPU : ", torch.cuda.get_device_name(0))
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")

    if torch.cuda.is_available():
        if args.gpu is not None:
            # Set specific GPU device if specified by args.gpu
            device = torch.device(f"cuda:{args.gpu}")
            model = model.to(device)
            print(
                f"Device is moved to GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}"
            )
        else:
            # Use DataParallel if no specific GPU is provided
            device = torch.device("cuda")
            model = torch.nn.DataParallel(model).to(device)
            num_gpus = torch.cuda.device_count()
            print(f"Number of GPUs available: {num_gpus}")
            for i in range(num_gpus):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print("Device set to all available GPU(s)")
    elif torch.backends.mps.is_available():
        # If MPS is available (for Apple Silicon)
        device = torch.device("mps")
        model = model.to(device)
        print("Device set to MPS")
    else:
        # Fallback to CPU if neither CUDA nor MPS is available
        device = torch.device("cpu")
        model = model.to(device)
        print("Device set to CPU. This will be slow.")

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = {}
    criterion["cls"] = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    if args.resume:
        if os.path.isfile(args.resume):
            try:
                print(f"=> Loading checkpoint '{args.resume}'")
                if args.gpu is None:
                    checkpoint = torch.load(args.resume, map_location=device)
                elif torch.cuda.is_available():
                    # Map model to be loaded to specified single GPU
                    loc = f"cuda:{args.gpu}"
                    checkpoint = torch.load(args.resume, map_location=loc)
                else:
                    raise RuntimeError(
                        "CUDA is not available, but a GPU checkpoint was specified."
                    )

                # Check for required keys in the checkpoint
                required_keys = [
                    "epoch",
                    "state_dict",
                    "optimizer",
                    "scheduler",
                    "best_acc1",
                ]
                for key in required_keys:
                    if key not in checkpoint:
                        raise KeyError(f"Missing key '{key}' in the checkpoint.")

                args.start_epoch = checkpoint["epoch"]
                best_acc1 = checkpoint["best_acc1"]

                if args.gpu is not None and torch.cuda.is_available():
                    best_acc1 = best_acc1.to(args.gpu)

                if not torch.cuda.is_available():
                    # Handle potential mismatches in the state_dict keys
                    state_dict = checkpoint["state_dict"]
                    model_state_dict = model.state_dict()
                    new_state_dict = {}

                    for key, value in state_dict.items():
                        if key.startswith("module."):
                            new_key = key[
                                len("module.") :
                            ]  # Remove "module." prefix if it exists
                        else:
                            new_key = key

                        # Only load matching keys
                        if new_key in model_state_dict:
                            new_state_dict[new_key] = value
                        else:
                            # print(f"=> Warning: Key '{new_key}' not found in the model. Skipping.")
                            raise ValueError(
                                f"Key '{new_key}' not found in the model. Cannot load checkpoint."
                            )

                    model.load_state_dict(new_state_dict)

                if torch.cuda.is_available():
                    model.load_state_dict(checkpoint["state_dict"])

                # Load optimizer and scheduler state dicts
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])

                print(
                    f"=> Successfully loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
                )

            except RuntimeError as e:
                print(f"=> Error loading checkpoint: {e}")
            except KeyError as e:
                print(f"=> Error: Checkpoint is missing required keys: {e}")
            except ValueError as e:
                print(f"=> Error: Missing Key- Value : {e}")
            except Exception as e:
                print(
                    f"=> An unexpected error occurred while loading the checkpoint: {e}"
                )

        else:
            print(f"=> No checkpoint found at '{args.resume}'")

    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")

    print("=> Using data from '{}'".format(args.data))

    # Define the transformation for light images

    # Do we need normalization for our case?

    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )

    # Training transformations
    train_transforms = transforms.Compose(
        [
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize,
        ]
    )

    val_transforms = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # normalize,
        ]
    )

    # Create the dataset
    train_dataset = PatchDataset_2(
        root=traindir,
        transform=train_transforms,
    )

    val_dataset = PatchDataset_2(
        root=valdir,
        transform=val_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
    )

    log_file = open("progress_log.txt", "w")

    if args.evaluate:
        validate(val_loader, model, criterion, device, args, log_file)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args, log_file)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device, args, log_file)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = {}
        is_best["acc1"] = acc1 >= best_acc1
        if is_best["acc1"]:
            log_line = (
                "[Acc@1] Best model found at epoch : "
                + str(epoch)
                + " | | Best Accuracy : "
                + str(acc1)
            )
            print(log_line)
            if log_file:
                log_file.write(log_line + "\n")
                log_file.flush()

        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
        )

    log_file.close()


def train(
    train_loader, model, criterion, optimizer, epoch, device, args, log_file=None
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    loss_CLS = AverageMeter("Loss_CLS", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_CLS, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
        log_file=log_file,
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (patches, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        patches = patches.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # batch_size X num_classes (logits)
        cls_output = model(patches)

        loss = {}
        loss["cls"] = criterion["cls"](cls_output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(cls_output, target, topk=(1, 5))
        # acc1, acc5 = (float('nan'), float('nan'))
        loss_CLS.update(loss["cls"].item(), patches.size(0))
        top1.update(acc1[0], patches.size(0))
        top5.update(acc5[0], patches.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        sum(loss[k] for k in loss).backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, device, args, log_file=None):

    def run_validate(loader, device, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (patches, target) in enumerate(loader):
                i = base_progress + i
                patches = patches.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output
                cls_output = model(patches)

                # measure loss
                loss = {}
                loss["cls"] = criterion["cls"](cls_output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(cls_output, target, topk=(1, 5))
                # acc1, acc5 = (float('nan'), float('nan'))
                loss_CLS.update(loss["cls"].item(), patches.size(0))
                top1.update(acc1[0], patches.size(0))
                top5.update(acc5[0], patches.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    loss_CLS = AverageMeter("Loss_CLS", ":.4e", Summary.AVERAGE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, loss_CLS, top1, top5],
        prefix="Test: ",
        log_file=log_file,
    )

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader, device)

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best["acc1"]:
        shutil.copyfile(filename, "model_best_acc1.pth.tar")


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_file=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_file = log_file

    def display(self, batch):
        # entries = [self.prefix + self.batch_fmtstr.format(batch)]
        # entries += [str(meter) for meter in self.meters]
        # print("\t".join(entries))
        # Generate the log string
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log_line = "\t".join(entries)

        # Print to console
        print(log_line)

        # Write to log file if available
        if self.log_file:
            self.log_file.write(log_line + "\n")
            self.log_file.flush()

    def display_summary(self):
        # entries = [" *"]
        # entries += [meter.summary() for meter in self.meters]
        # print(" ".join(entries))

        # Generate the summary string
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        log_line = " ".join(entries)

        # Print to console
        print(log_line)

        # Write to log file if available
        if self.log_file:
            self.log_file.write(log_line + "\n")
            self.log_file.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
