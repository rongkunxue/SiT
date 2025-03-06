import os
import torch
from torch import nn
from accelerate import Accelerator
from torchvision.utils import save_image
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from download import find_model
from models import SiT_XL_2
from PIL import Image
import torchvision
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")
import logging

import json
import torch.utils.data as data
import numpy as np
import sys

@torch.no_grad()
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

@torch.no_grad()
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



def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, "_is_replica", False):

        def find_tensor_attributes(module):
            tuples = [
                (k, v)
                for k, v in module.__dict__.items()
                if torch.is_tensor(v) and v.requires_grad
            ]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())

class TinyImageNet(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if self.Train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, "r") as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, "r") as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [
                d
                for d in os.listdir(self.train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            ]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [
                d
                for d in os.listdir(val_image_dir)
                if os.path.isfile(os.path.join(train_dir, d))
            ]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, "r") as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (
                                path,
                                self.class_to_tgt_idx[self.val_img_to_class[fname]],
                            )
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, "rb") as f:
            sample = Image.open(img_path)
            sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging

from models import SiT_models
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb

class classifyHead(nn.Module):
    def __init__(self):
        super(classifyHead, self).__init__()
        self.mypool=nn.Sequential(
            nn.Flatten(),
        )
        self.classifier=nn.Sequential(
            nn.LayerNorm(4096),
            nn.Linear(4096, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1000, bias=False),
        )

    def forward(self, x):
        x = self.mypool(x)
        x = self.classifier(x)
        return x


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    accelerator = Accelerator()

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)

        logging.info(f"Experiment directory created at {experiment_dir}")
        wandb.init(project="SiT", name=experiment_name, config=vars(args)) 


    accelerator.wait_for_everyone()
    
    device = accelerator.device

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    head = classifyHead()

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logging.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW([{'params': model.parameters()}, {'params': head.parameters()}], lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    train_dataset = TinyImageNet("/root/Github/data/tiny-imagenet-200", True, transform)
    # use cifar10 for training
    train_cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    eval_cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    eval_dataset = TinyImageNet("/root/Github/data/tiny-imagenet-200", True, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.global_batch_size,
        shuffle=False,
        num_workers=args.num_workers)
    train_cifar_dataloader=DataLoader(train_cifar_dataset, batch_size=args.global_batch_size, shuffle=True, num_workers=args.num_workers)
    eval_cifar_dataloader=DataLoader(eval_cifar_dataset, batch_size=args.global_batch_size, shuffle=False, num_workers=args.num_workers)    

    model_param = find_parameters(model)

    model, head, opt, train_loader,eval_loader,train_cifar_dataloader,eval_cifar_dataloader = accelerator.prepare(model, head, opt, train_loader,eval_loader,train_cifar_dataloader,eval_cifar_dataloader)

    model.train() 
    
    logging.info(f"Training for {args.epochs} epochs...")
    from timm.loss import LabelSmoothingCrossEntropy
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    num_sampling_steps = 32 #@param {type:"slider", min:0, max:1000, step:1}
    ODE_sampling_method = "euler" #@param ["dopri5", "euler", "rk4"]
    atol = 1e-6
    rtol = 1e-3
    ode_sampler = Sampler(transport)
    sample_fn = ode_sampler.reverse_sample_ode(
        sampling_method=ODE_sampling_method,
        atol=atol,
        rtol=rtol,
        num_steps=num_sampling_steps,
        with_grad=True,
    )
    for epoch in range(args.epochs):
        logging.info(f"Beginning epoch {epoch}...")
        # if epoch % 5 == 0:
        #     top1 = AverageMeter("Acc@1", ":6.2f")
        #     top5 = AverageMeter("Acc@5", ":6.2f")
        #     model.eval()
        #     with torch.no_grad():
        #         for x, y in eval_loader:
        #             y_null = torch.tensor([1000] * x.shape[0], device=x.device)
        #             sample_model_kwargs = dict(y=y_null)
        #             x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        #             samples = sample_fn(x, model.forward, model_param, **sample_model_kwargs)[-1]
        #             logit = head(samples)
        #             logit,y=accelerator.gather_for_metrics((logit,y))
        #             acc1, acc5 = accuracy(logit,y, topk=(1, 5))
        #             top1.update(acc1[0], logit.size(0))
        #             top5.update(acc5[0], logit.size(0))
        #             if accelerator.is_main_process:
        #                 wandb.log(
        #                     {
        #                         f"eval/tiny_acc1": top1.avg,
        #                         f"eval/tiny_acc5": top5.avg,
        #                         f"eval/tiny_epoch": epoch,
        #                     },
        #                     commit=False,
        #                 )

        # if epoch % 5 == 0:
        #     top1 = AverageMeter("Acc@1", ":6.2f")
        #     top5 = AverageMeter("Acc@5", ":6.2f")
        #     model.eval()
        #     with torch.no_grad():
        #         for x, y in eval_cifar_dataloader:
        #             y_null = torch.tensor([1000] * x.shape[0], device=x.device)
        #             sample_model_kwargs = dict(y=y_null)
        #             x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        #             samples = sample_fn(x, model.forward, model_param, **sample_model_kwargs)[-1]
        #             logit = head(samples)
        #             logit,y=accelerator.gather_for_metrics((logit,y))
        #             acc1, acc5 = accuracy(logit,y, topk=(1, 5))
        #             top1.update(acc1[0], logit.size(0))
        #             top5.update(acc5[0], logit.size(0))
        #             if accelerator.is_main_process:
        #                 wandb.log(
        #                     {
        #                         f"eval/cifar_acc1": top1.avg,
        #                         f"eval/cifar_acc5": top5.avg,
        #                         f"eval/cifar_epoch": epoch,
        #                     },
        #                     commit=False,
        #                 )
                        
        for x, y in train_cifar_dataloader:
            y_null = torch.tensor([1000] * x.shape[0], device=x.device)
            sample_model_kwargs = dict(y=y_null)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            samples = sample_fn(x, model.forward, model_param, **sample_model_kwargs)[-1] 
            logit = head(samples) 
            loss = criterion(logit, y)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            if accelerator.is_main_process:
                wandb.log(
                    {
                        f"train/cifar_loss": loss.item(),
                        f"train/cifar_epoch": epoch,
                    },
                    commit=False,
                )
                 
        for x, y in train_loader:
            y_null = torch.tensor([1000] * x.shape[0], device=x.device)
            sample_model_kwargs = dict(y=y_null)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            samples = sample_fn(x, model.forward, model_param, **sample_model_kwargs)[-1] 
            logit = head(samples) 
            loss = criterion(logit, y)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            if accelerator.is_main_process:
                wandb.log(
                    {
                        f"train/tiny_loss": loss.item(),
                        f"train/tiny_epoch": epoch,
                    },
                    commit=False,
                )

    
        
                        

if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default="SiT-XL-2-256x256.pt",
                        help="Optional path to a custom SiT checkpoint")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)

