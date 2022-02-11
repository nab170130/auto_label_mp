# -*- coding: utf-8 -*-
"""
# SETUP

## Repo Installation
"""

import sys

sys.path.append("auto_label_mp")
sys.path.append("distil")

"""## Imports"""

import copy
import csv
import json
import math
import numpy as np
import os
import pickle
import sys
import time
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import zipfile


from argparse import ArgumentParser
from smi_autolabeler import PartitionStrategy, SMIAutoLabeler

from distil.active_learning_strategies.badge import BADGE
from distil.active_learning_strategies.entropy_sampling import EntropySampling
from distil.active_learning_strategies.random_sampling import RandomSampling
from distil.active_learning_strategies.partition_strategy import PartitionStrategy as ALPartitionStrategy
from distil.active_learning_strategies.strategy import Strategy
from distil.utils.models import MnistNet, ResNet18
from distil.utils.utils import LabeledToUnlabeledDataset

from PIL import Image

from scipy.io import loadmat

from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision._internally_replaced_utils import load_state_dict_from_url

from typing import Type, Any, Callable, Union, List, Optional


"""# EXPERIMENTS

## Definitions

### Checkpointing
"""

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, last=False, freeze=False) -> Tensor:
        # See note [TorchScript super()]
        if freeze:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        out = self.fc(x)
        if last:
            return out, x
        else:
            return out

    def forward(self, x: Tensor, last=False, freeze=False) -> Tensor:
        return self._forward_impl(x, last, freeze)

    def get_embedding_dim(self):
        return self.fc.in_features

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet18-f37072fd.pth", progress=progress)
        model.load_state_dict(state_dict)

    return model


def VariableSizeResNet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)        

class Checkpoint:

    def __init__(self, exp_dict=None, idx_bit_vector=None, labels=None, state_dict=None, experiment_name=None, path=None):

        # If a path is supplied, load a checkpoint from there.
        if path is not None:

            if experiment_name is not None:
                self.load_checkpoint(path, experiment_name)
            else:
                raise ValueError("Checkpoint contains None value for experiment_name")

            return

        if exp_dict is None:
            raise ValueError("Checkpoint contains None value for acc_list")

        if idx_bit_vector is None:
            raise ValueError("Checkpoint contains None value for idx_bit_vector")

        if state_dict is None:
            raise ValueError("Checkpoint contains None value for state_dict")

        if labels is None:
            raise ValueError("Checkpoint contains None value for labels")

        if experiment_name is None:
            raise ValueError("Checkpoint contains None value for experiment_name")

        self.exp_dict = exp_dict
        self.idx_bit_vector = idx_bit_vector
        self.labels = labels
        self.state_dict = state_dict
        self.experiment_name = experiment_name

    def __eq__(self, other):

        # Check if the accuracy lists are equal
        acc_lists_equal = self.exp_dict == other.exp_dict

        # Check if the indices are equal
        indices_equal = self.idx_bit_vector == other.idx_bit_vector

        # Check if the labels are equal
        labels_equal = self.labels == other.labels

        # Check if the experiment names are equal
        experiment_names_equal = self.experiment_name == other.experiment_name

        return acc_lists_equal and indices_equal and labels_equal and experiment_names_equal

    def save_checkpoint(self, path):

        # Get current time to use in file timestamp
        timestamp = time.time_ns()

        # Create the path supplied
        os.makedirs(path, exist_ok=True)

        # Name saved files using timestamp to add recency information
        save_path = os.path.join(path, F"c{timestamp}1")

        # Write this checkpoint to the save location
        with open(save_path, 'wb') as save_file:
            pickle.dump(self, save_file)

    def load_checkpoint(self, path, experiment_name):

        # Obtain a list of all files present at the path
        timestamp_save_no = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        # If there are no such files, set values to None and return
        if len(timestamp_save_no) == 0:
            self.exp_dict = None
            self.idx_bit_vector = None
            self.labels = None
            self.state_dict = None
            return

        # Sort the list of strings to get the most recent
        timestamp_save_no.sort(reverse=True)

        while len(timestamp_save_no) >= 1:

            load_file = timestamp_save_no.pop(0)

            # Form the paths to the files
            load_path = os.path.join(path, load_file)

            # Load the checkpoint
            with open(load_path, 'rb') as load_file:
                checkpoint = pickle.load(load_file)

            F"{dataset}_{al_strategy}_{human_correct_strategy}_{auto_assign_strategy}_{b1}_{b2}_{b3}_{seed_size}_{rounds}_{run_count}_{adaptive}.json"
            
            checkpoint_exp_name = checkpoint.experiment_name
            
            # Get filename fields
            if "highest_confidence" in checkpoint_exp_name:
                checkpoint_exp_name_field_array = checkpoint_exp_name.split(".")[0].split("_highest_confidence_")
                checkpoint_exp_name_fields = checkpoint_exp_name_field_array[0].split("_") + ["highest_confidence"] + checkpoint_exp_name_field_array[1].split("_")
            else:
                checkpoint_exp_name_fields = checkpoint_exp_name.split(".")[0].split("_")
            
            # Get filename fields
            if "highest_confidence" in experiment_name:
                exp_name_field_array = experiment_name.split(".")[0].split("_highest_confidence_")
                exp_name_fields = exp_name_field_array[0].split("_") + ["highest_confidence"] + exp_name_field_array[1].split("_")
            else:
                exp_name_fields = experiment_name.split(".")[0].split("_")
            
            # Compare dataset, al strategy, human-correct strategy, auto-assign strategy, b1, b2, b3, seed size, adaptive
            num_fields = 11
            should_skip = False
            ignore_field_indices = [8,9]
            for field_counter in range(num_fields):
                
                # DO NOT CHECK THE IGNORED FIELDS
                if field_counter in ignore_field_indices:
                    continue
                
                if checkpoint_exp_name_fields[field_counter] != exp_name_fields[field_counter]:
                    should_skip = True
                    break
                
            if should_skip:
                continue

            # This checkpoint will suffice. Populate this checkpoint's fields 
            # with the selected checkpoint's fields.
            self.exp_dict = checkpoint.exp_dict
            self.idx_bit_vector = checkpoint.idx_bit_vector
            self.labels = checkpoint.labels
            self.state_dict = checkpoint.state_dict
            return

        # Instantiate None values in acc_list, indices, and model
        self.exp_dict = None
        self.idx_bit_vector = None
        self.labels = None
        self.state_dict = None

    def get_saved_values(self):

        return (self.exp_dict, self.idx_bit_vector, self.labels, self.state_dict)

def delete_checkpoints(checkpoint_directory, experiment_name):

    # Iteratively go through each checkpoint, deleting those whose experiment name matches.
    timestamp_save_no = [f for f in os.listdir(checkpoint_directory) if os.path.isfile(os.path.join(checkpoint_directory, f))]

    for file in timestamp_save_no:

        delete_file = False

        # Get file location
        file_path = os.path.join(checkpoint_directory, file)

        if not os.path.exists(file_path):
            continue

        # Unpickle the checkpoint and see if its experiment name matches
        with open(file_path, "rb") as load_file:

            checkpoint_copy = pickle.load(load_file)
            
            # Compare dataset, al strategy, human-correct strategy, auto-assign strategy, b1, b2, b3, seed size, adaptive
            checkpoint_exp_name = checkpoint_copy.experiment_name
            
            # Get filename fields
            if "highest_confidence" in checkpoint_exp_name:
                checkpoint_exp_name_field_array = checkpoint_exp_name.split(".")[0].split("_highest_confidence_")
                checkpoint_exp_name_fields = checkpoint_exp_name_field_array[0].split("_") + ["highest_confidence"] + checkpoint_exp_name_field_array[1].split("_")
            else:
                checkpoint_exp_name_fields = checkpoint_exp_name.split(".")[0].split("_")
            
            # Get filename fields
            if "highest_confidence" in experiment_name:
                exp_name_field_array = experiment_name.split(".")[0].split("_highest_confidence_")
                exp_name_fields = exp_name_field_array[0].split("_") + ["highest_confidence"] + exp_name_field_array[1].split("_")
            else:
                exp_name_fields = experiment_name.split(".")[0].split("_")
            
            num_fields = 11
            delete_file = True
            ignore_field_indices = [8,9]
            for field_counter in range(num_fields):
                
                # DO NOT CHECK THE IGNORED FIELDS
                if field_counter in ignore_field_indices:
                    continue
                
                if checkpoint_exp_name_fields[field_counter] != exp_name_fields[field_counter]:
                    delete_file = False
                    break
        
        # Delete this file only if the experiment name matched
        if delete_file:
            os.remove(file_path)

"""### Evaluation Utilities"""

def get_label_counts(dataset, nclasses, batch_size=64):

    label_counts = [0 for x in range(nclasses)]
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            for cls in range(nclasses):
                count = len(torch.where(labels==cls)[0])
                label_counts[cls] += count

    return label_counts

def get_labels(dataset, batch_size=64):

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            all_labels.extend(labels)

    return torch.tensor(all_labels)

"""### Replace Label Dataset"""

class ReplaceLabelDataset(Dataset):

    def __init__(self, labeled_dataset, new_label_sequence):
        self.labeled_dataset = labeled_dataset
        self.new_label_sequence = new_label_sequence

    def __getitem__(self, index):
        data, old_index = self.labeled_dataset[index]
        new_index = self.new_label_sequence[index]
        return data, new_index

    def __len__(self):
        return len(self.new_label_sequence)

"""### Selection Utilities"""

def get_class_subset(dataset, class_to_retrieve, batch_size=64):

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    subset_idxs = []
    eval_idxs = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):

            matching_class_batch_idxs = torch.where(labels==class_to_retrieve)[0]
            matching_class_batch_idxs = matching_class_batch_idxs + eval_idxs
            subset_idxs.extend(matching_class_batch_idxs)
            eval_idxs += len(labels)

    return Subset(dataset, subset_idxs)

def label_new_points(unlabeled_dataset, to_add_idx_class_list, selection_mode):
    if selection_mode == "auto":
        new_labels_list = [label for (_,label) in to_add_idx_class_list]
        selected_idx = [index for (index,_) in to_add_idx_class_list]
        return ReplaceLabelDataset(Subset(unlabeled_dataset, selected_idx), new_labels_list)
    elif selection_mode == "hil":
        selected_idx = [index for (index,_) in to_add_idx_class_list]
        return Subset(unlabeled_dataset, selected_idx)

"""### Default Training Class"""

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class AddIndexDataset(Dataset):
    
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset
        
    def __getitem__(self, index):
        data, label = self.wrapped_dataset[index]
        return data, label, index
    
    def __len__(self):
        return len(self.wrapped_dataset)

#custom training
class data_train:

    """
    Provides a configurable training loop for AL.
    
    Parameters
    ----------
    training_dataset: torch.utils.data.Dataset
        The training dataset to use
    net: torch.nn.Module
        The model to train
    args: dict
        Additional arguments to control the training loop
        
        `batch_size` - The size of each training batch (int, optional)
        `islogs`- Whether to return training metadata (bool, optional)
        `optimizer`- The choice of optimizer. Must be one of 'sgd' or 'adam' (string, optional)
        `isverbose`- Whether to print more messages about the training (bool, optional)
        `isreset`- Whether to reset the model before training (bool, optional)
        `max_accuracy`- The training accuracy cutoff by which to stop training (float, optional)
        `min_diff_acc`- The minimum difference in accuracy to measure in the window of monitored accuracies. If all differences are less than the minimum, stop training (float, optional)
        `window_size`- The size of the window for monitoring accuracies. If all differences are less than 'min_diff_acc', then stop training (int, optional)
        `criterion`- The criterion to use for training (typing.Callable[], optional)
        `device`- The device to use for training (string, optional)
    """
    
    def __init__(self, training_dataset, net, args):

        self.training_dataset = AddIndexDataset(training_dataset)
        self.net = net
        self.args = args
        
        self.n_pool = len(training_dataset)
        
        if 'islogs' not in args:
            self.args['islogs'] = False

        if 'optimizer' not in args:
            self.args['optimizer'] = 'sgd'
        
        if 'isverbose' not in args:
            self.args['isverbose'] = False
        
        if 'isreset' not in args:
            self.args['isreset'] = True

        if 'max_accuracy' not in args:
            self.args['max_accuracy'] = 0.95

        if 'min_diff_acc' not in args: #Threshold to monitor for
            self.args['min_diff_acc'] = 0.001

        if 'window_size' not in args:  #Window for monitoring accuracies
            self.args['window_size'] = 10
            
        if 'criterion' not in args:
            self.args['criterion'] = nn.CrossEntropyLoss()
            
        if 'should_freeze' not in args:
            self.args['should_freeze'] = False
            
        if 'device' not in args:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args['device']

    def update_index(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def update_data(self, new_training_dataset):
        """
        Updates the training dataset with the provided new training dataset
        
        Parameters
        ----------
        new_training_dataset: torch.utils.data.Dataset
            The new training dataset
        """
        self.training_dataset = AddIndexDataset(new_training_dataset)

    def get_acc_on_set(self, test_dataset):
        
        """
        Calculates and returns the accuracy on the given dataset to test
        
        Parameters
        ----------
        test_dataset: torch.utils.data.Dataset
            The dataset to test
        Returns
        -------
        accFinal: float
            The fraction of data points whose predictions by the current model match their targets
        """	
        
        try:
            self.clf
        except:
            self.clf = self.net

        if test_dataset is None:
            raise ValueError("Test data not present")
        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1 
        
        loader_te = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)
        self.clf.eval()
        accFinal = 0.

        with torch.no_grad():        
            self.clf = self.clf.to(device=self.device)
            for batch_id, (x,y) in enumerate(loader_te):     
                x, y = x.to(device=self.device), y.to(device=self.device)
                out = self.clf(x)
                accFinal += torch.sum(1.0*(torch.max(out,1)[1] == y)).item() #.data.item()

        return accFinal / len(test_dataset)

    def _train_weighted(self, epoch, loader_tr, optimizer, gradient_weights):
        self.clf.train()
        accFinal = 0.
        criterion = self.args['criterion']
        criterion.reduction = "none"

        for batch_id, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(device=self.device), y.to(device=self.device)
            gradient_weights = gradient_weights.to(device=self.device)

            optimizer.zero_grad()
            out = self.clf(x)

            # Modify the loss function to apply weights before reducing to a mean
            loss = criterion(out, y.long())

            # Perform a dot product with the loss vector and the weight vector, then divide by batch size.
            weighted_loss = torch.dot(loss, gradient_weights[idxs])
            weighted_loss = torch.div(weighted_loss, len(idxs))

            accFinal += torch.sum(torch.eq(torch.max(out,1)[1],y)).item() #.data.item()

            # Backward now does so on the weighted loss, not the regular mean loss
            weighted_loss.backward() 

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        return accFinal / len(loader_tr.dataset), weighted_loss

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        accFinal = 0.
        criterion = self.args['criterion']
        criterion.reduction = "mean"

        for batch_id, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(device=self.device), y.to(device=self.device)

            optimizer.zero_grad()
            
            out = self.clf(x, freeze=self.args['should_freeze'])
            loss = criterion(out, y.long())
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).item()
            loss.backward()

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        return accFinal / len(loader_tr.dataset), loss

    def check_saturation(self, acc_monitor):
        
        saturate = True

        for i in range(len(acc_monitor)):
            for j in range(i+1, len(acc_monitor)):
                if acc_monitor[j] - acc_monitor[i] >= self.args['min_diff_acc']:
                    saturate = False
                    break

        return saturate

    def train(self, gradient_weights=None):

        """
        Initiates the training loop.
        
        Parameters
        ----------
        gradient_weights: list, optional
            The weight of each data point's effect on the loss gradient. If none, regular training will commence. If not, weighted training will commence.
        Returns
        -------
        model: torch.nn.Module
            The trained model. Alternatively, this will also return the training logs if 'islogs' is set to true.
        """        

        print('Training..')
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        train_logs = []
        n_epoch = self.args['n_epoch']
        
        if self.args['isreset']:
            if 'pretrained_class_count' in self.args:
                
                # Get pretrained resnet18 but reset its fc layer.
                pt_resnet = VariableSizeResNet18(pretrained=True)
                last_layer_features = pt_resnet.fc.in_features
                pt_resnet.fc = nn.Linear(last_layer_features, self.args['pretrained_class_count'])
                self.clf = pt_resnet.to(device=self.device)
            else:
                self.clf = self.net.apply(weight_reset).to(device=self.device)
        else:
            try:
                self.clf
            except:
                self.clf = self.net.apply(weight_reset).to(device=self.device)

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
            lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
        
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1

        # Set shuffle to true to encourage stochastic behavior for SGD
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        epoch = 1
        accCurrent = 0
        is_saturated = False
        acc_monitor = []

        while (accCurrent < self.args['max_accuracy']) and (epoch < n_epoch) and (not is_saturated): 
            
            if gradient_weights is None:
                accCurrent, lossCurrent = self._train(epoch, loader_tr, optimizer)
            else:
                accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights)
            
            acc_monitor.append(accCurrent)

            if self.args['optimizer'] == 'sgd':
                lr_sched.step()
            
            epoch += 1
            if(self.args['isverbose']):
                if epoch % 50 == 0:
                    print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)

            #Stop training if not converging
            if len(acc_monitor) >= self.args['window_size']:

                is_saturated = self.check_saturation(acc_monitor)
                del acc_monitor[0]

            log_string = 'Epoch:' + str(epoch) + '- training accuracy:'+str(accCurrent)+'- training loss:'+str(lossCurrent)
            train_logs.append(log_string)
            if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                self.clf = self.net.apply(weight_reset).to(device=self.device)
                
                if self.args['optimizer'] == 'sgd':

                    optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                    lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

                else:
                    optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        print('Epoch:', str(epoch), 'Training accuracy:', round(accCurrent, 3), flush=True)

        if self.args['islogs']:
            return self.clf, train_logs
        else:
            return self.clf

class ConfidenceAutoLabeler(Strategy):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}): #
        
        super(ConfidenceAutoLabeler, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)

    def select(self, budget):
        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	

        self.model.eval()       

        # Get the model's predictions
        probs = self.predict_prob(self.unlabeled_dataset)
        max_class_prob, class_predictions = torch.max(probs, dim=1)
        sorted_max_class_prob, sorted_max_class_indices = torch.sort(max_class_prob, descending=True)
        sorted_class_predictions = class_predictions[sorted_max_class_indices]

        budgets_to_use = [(budget * i) // self.target_classes for i in range(self.target_classes + 1)]
        selected_idx = []

        for sel_class in range(self.target_classes):

            class_budget = budgets_to_use[sel_class + 1] - budgets_to_use[sel_class]
            sel_class_idx = torch.where(sorted_class_predictions == sel_class)[0]
            sel_class_to_label_idx = (sorted_max_class_indices[sel_class_idx])[:class_budget]
            sel_class_to_label_prob = (sorted_max_class_prob[sel_class_idx])[:class_budget]
            sel_class_list = zip(sel_class_to_label_idx.tolist(), sel_class_to_label_prob.tolist())
            selected_idx.append(sel_class_list)

        return selected_idx

"""### Training Loop"""

def al_train_loop(full_dataset, train_lake_usage_list, test_dataset, net, n_rounds, b1, b2, b3, args, nclasses, active_learning_name, human_correct_name, auto_assign_name, checkpoint_directory, experiment_name, adaptive=False):

    # Get all labels in the full dataset as the initial assigned labels
    assigned_labels = get_labels(full_dataset, args['batch_size'])

    # Define initial experiment dictionary
    exp_dict = {'set_sizes':[],
                'test_accuracies':[],
                'b1':b1,
                'b2':b2,
                'b3':b3,
                'auto_assign_strategy': auto_assign_name,
                'human_correct_strategy': human_correct_name,
                'active_learning_strategy': active_learning_name,
                'auto_assigned_selected_idx': [],
                'human_corrected_selected_idx': [],
                'active_learning_selected_idx': [],
                'auto_assigned_selection_matrices': [],
                'human_corrected_selection_matrices': [],
                'auto_assign_selection_times': [],
                'human_correct_selection_times': [],
                'al_selection_times': [],
                'train_times': []
                }

    # Set the initial round to 1
    initial_round = 1

    # Obtain a checkpoint if one exists
    training_checkpoint = Checkpoint(experiment_name=experiment_name, path=checkpoint_directory)
    rec_exp_dict, rec_train_lake_usage_list, rec_labels, rec_state_dict = training_checkpoint.get_saved_values()

    # Check if there are values to recover
    if rec_exp_dict is not None:

        # Restore the experiment dict
        exp_dict = rec_exp_dict

        # Restore the train-lake usage list
        train_lake_usage_list = rec_train_lake_usage_list

        # Restore the auto-assigned labels
        assigned_labels = rec_labels

        # Restore the model
        net.load_state_dict(rec_state_dict)

        # Fix the initial round
        initial_round = len(exp_dict['set_sizes'])

    # Ensure the loaded model is moved to the right device
    net = net.to(args['device'])

    # Obtain the labeled/lake datasets
    train_indices = [i for (i,x) in enumerate(train_lake_usage_list) if x == 1]
    train_dataset = Subset(ReplaceLabelDataset(full_dataset, assigned_labels), train_indices)
    lake_indices = [i for (i,x) in enumerate(train_lake_usage_list) if x == 0]
    lake_dataset = Subset(full_dataset, lake_indices)

    # Initialize the training helper
    dt = data_train(train_dataset, net, args)  

    # Get information about the initial model this is the first round
    if initial_round == 1:
        initial_test_acc = dt.get_acc_on_set(test_dataset)
        exp_dict['test_accuracies'].append(initial_test_acc)
        exp_dict['set_sizes'].append(len(train_dataset))
        print("Initial Test Accuracy:", round(initial_test_acc*100, 2), flush=True)

    # Initialize the AL strategy.
    if active_learning_name == "badge":
        strat_args = copy.deepcopy(args)
        strat_args['num_partitions'] = strat_args['num_partitions_al']
        strat_args['wrapped_strategy_class'] = BADGE
        active_learning_strategy = ALPartitionStrategy(train_dataset, LabeledToUnlabeledDataset(lake_dataset), net, nclasses, strat_args)
    elif active_learning_name == "entropy":
        strat_args = copy.deepcopy(args)
        active_learning_strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(lake_dataset), net, nclasses, strat_args)
    elif active_learning_name == "random":
        strat_args = copy.deepcopy(args)
        active_learning_strategy = RandomSampling(train_dataset, LabeledToUnlabeledDataset(lake_dataset), net, nclasses, strat_args)
    else:
        raise ValueError("al_strategy should take one of ['badge', 'entropy, 'random']")

    # Initialize the auto-assign strategy.
    if auto_assign_name == "fl2mi":
        strat_args = copy.deepcopy(args)
        strat_args['optimizer'] = "LazyGreedy"
        strat_args['wrapped_strategy_class'] = SMIAutoLabeler
        strat_args['smi_function'] = 'fl2mi'
        strat_args['num_partitions'] = strat_args['num_partitions_auto']
        auto_assign_strategy = PartitionStrategy(train_dataset, LabeledToUnlabeledDataset(lake_dataset), net, nclasses, strat_args, query_dataset=None)
    elif auto_assign_name == "gcmi":
        strat_args = copy.deepcopy(args)
        strat_args['optimizer'] = "LazyGreedy"
        strat_args['wrapped_strategy_class'] = SMIAutoLabeler
        strat_args['smi_function'] = 'gcmi'
        strat_args['num_partitions'] = strat_args['num_partitions_auto']
        auto_assign_strategy = PartitionStrategy(train_dataset, LabeledToUnlabeledDataset(lake_dataset), net, nclasses, strat_args, query_dataset=None)
    elif auto_assign_name == "logdetmi":
        strat_args = copy.deepcopy(args)
        strat_args['optimizer'] = "LazyGreedy"
        strat_args['wrapped_strategy_class'] = SMIAutoLabeler
        strat_args['smi_function'] = 'logdetmi'
        strat_args['num_partitions'] = strat_args['num_partitions_auto']
        auto_assign_strategy = PartitionStrategy(train_dataset, LabeledToUnlabeledDataset(lake_dataset), net, nclasses, strat_args, query_dataset=None)
    elif auto_assign_name == "highest_confidence":
        strat_args = copy.deepcopy(args)
        auto_assign_strategy = ConfidenceAutoLabeler(train_dataset, LabeledToUnlabeledDataset(lake_dataset), net, nclasses, strat_args)
    else:
        raise ValueError("auto_assign_strategy should take one of ['fl2mi', 'gcmi', 'logdetmi', 'highest_confidence']")

    # Initialize the human-correct strategy.
    if human_correct_name == "fl1mi":
        strat_args = copy.deepcopy(args)
        strat_args['optimizer'] = "LazyGreedy"
        strat_args['wrapped_strategy_class'] = SMIAutoLabeler
        strat_args['smi_function'] = 'fl1mi'
        strat_args['num_partitions'] = strat_args['num_partitions_human']
        human_correct_strategy = PartitionStrategy(train_dataset, LabeledToUnlabeledDataset(lake_dataset), net, nclasses, strat_args, query_dataset=None)
    elif human_correct_name == "logdetmi":
        strat_args = copy.deepcopy(args)
        strat_args['optimizer'] = "LazyGreedy"
        strat_args['wrapped_strategy_class'] = SMIAutoLabeler
        strat_args['smi_function'] = 'logdetmi'
        strat_args['num_partitions'] = strat_args['num_partitions_human']
        human_correct_strategy = PartitionStrategy(train_dataset, LabeledToUnlabeledDataset(lake_dataset), net, nclasses, strat_args, query_dataset=None)
    else:
        raise ValueError("human_correct_strategy should take one of ['fl1mi', 'logdetmi']")

    # Record the training transform and test transform for disabling purposes
    train_transform = full_dataset.transform
    test_transform = test_dataset.transform

    end_b3 = b3
    b_tot = b2 + b3

    # Begin AL loop
    for rd in range(initial_round, n_rounds+1):

        print('-------------------------------------------------')
        print('Round', rd) 
        print('-------------------------------------------------')

        if adaptive:
            
            round_fraction = (rd - 1) / (n_rounds - 1)
            b2 = math.floor(b_tot - round_fraction * end_b3)
            b3 = math.floor(round_fraction * end_b3)

        start_auto_time = time.time()
        full_dataset.transform = test_transform # Disable any augmentation while selecting points

        # ==============================
        # Auto-Assign Selection
        # ==============================

        # If the budget for doing per-class targeted selection is 0, simply assign empty lists for selected idx.
        # Otherwise, proceed with the selection.
        if b3 == 0:
            auto_assigned_selected_idx = [[] for x in range(nclasses)]
        else:

            # Calculate idx with respect to lake set to label. Make sure to use current training set as the query set
            auto_assign_strategy.update_queries(train_dataset)
            auto_assigned_selected_idx = auto_assign_strategy.select(b3)

            # Convert to full dataset indices
            for i, selected_idx_per_class in enumerate(auto_assigned_selected_idx):
                auto_assigned_selected_idx[i] = [(lake_indices[j], associated_gain) for (j, associated_gain) in selected_idx_per_class]

            # Update the labels for auto-labeled points
            for sel_class, selected_idx_per_class in enumerate(auto_assigned_selected_idx):
                for (selected_index, _) in selected_idx_per_class:
                    assigned_labels[selected_index] = sel_class

            # Update lake/train indices and lake_train_usage_list
            for selected_idx_per_class in auto_assigned_selected_idx:
                for (selected_index, _) in selected_idx_per_class:
                    train_lake_usage_list[selected_index] = 1

            train_indices = [i for (i,x) in enumerate(train_lake_usage_list) if x == 1]
            lake_indices = [i for (i,x) in enumerate(train_lake_usage_list) if x == 0]

            # Now, update the train dataset and the lake dataset
            train_dataset = Subset(ReplaceLabelDataset(full_dataset, assigned_labels), train_indices)
            lake_dataset = Subset(full_dataset, lake_indices)

        start_human_time = time.time()
        
        # ==========================
        # HUMAN-CORRECT SELECTION
        # ==========================
        
        # If the budget for doing per-class targeted selection is 0, simply assign empty lists for selected idx.
        # Otherwise, proceed with the selection.
        if b2 == 0:
            human_corrected_selected_idx = [[] for x in range(nclasses)]
        else:

            # Calculate idx with respect to lake set to label. Make sure to use current training set as the query set and 
            # current unlabeled set as the unlabeled set.
            human_correct_strategy.update_queries(train_dataset)
            human_correct_strategy.update_data(train_dataset, LabeledToUnlabeledDataset(lake_dataset))
            human_corrected_selected_idx = human_correct_strategy.select(b2)

            # Convert to full dataset indices
            for i, selected_idx_per_class in enumerate(human_corrected_selected_idx):
                human_corrected_selected_idx[i] = [(lake_indices[j], associated_gain) for (j, associated_gain) in selected_idx_per_class]

            # Note that the labels are not updated; they are assigned by the human correctly.

            # Update lake/train indices and lake_train_usage_list
            for selected_idx_per_class in human_corrected_selected_idx:
                for (selected_index, _) in selected_idx_per_class:
                    train_lake_usage_list[selected_index] = 1

            train_indices = [i for (i,x) in enumerate(train_lake_usage_list) if x == 1]
            lake_indices = [i for (i,x) in enumerate(train_lake_usage_list) if x == 0]

            # Now, update the train dataset and the lake dataset
            train_dataset = Subset(ReplaceLabelDataset(full_dataset, assigned_labels), train_indices)
            lake_dataset = Subset(full_dataset, lake_indices)

        start_al_time = time.time()

        # =======================
        # BEGIN PURE AL SELECTION
        # =======================

        # See if AL selection needs to be done. If not, assign blank lists as before.
        if b1 == 0:
            selected_idx = []
        else:
            # ---- Get Selected Idx WRPT Full Dataset ----

            # First, be sure to update the active learning strategy with the new unlabeled dataset.
            active_learning_strategy.update_data(train_dataset, LabeledToUnlabeledDataset(lake_dataset))

            # Now, select the pure AL points.
            selected_idx = active_learning_strategy.select(b1)

            # selected_idx, unlike before, is flat with no scores. We simply map back to the full dataset.
            selected_idx = [lake_indices[j] for j in selected_idx]

            # Predict the labels of the selected points under the current model. These will be stored alongside the AL selected idx.
            selected_subset = LabeledToUnlabeledDataset(Subset(full_dataset, selected_idx))
            subset_predicted_classes = list(active_learning_strategy.predict(selected_subset).cpu().numpy().tolist())

            # ----- Label These Points ----

            # As before, we already have the ground-truth labels in this experiment; we do not 
            # need to take action for the human-corrected points. They will already have 
            # the correct label when assigned to the training dataset.

            # Update lake/train indices and lake_train_usage_list
            for selected_index in selected_idx:
                train_lake_usage_list[selected_index] = 1

            train_indices = [i for (i,x) in enumerate(train_lake_usage_list) if x == 1]
            lake_indices = [i for (i,x) in enumerate(train_lake_usage_list) if x == 0]

            # Now, update the train dataset and the lake dataset
            train_dataset = Subset(ReplaceLabelDataset(full_dataset, assigned_labels), train_indices)
            lake_dataset = Subset(full_dataset, lake_indices)
            
            # Do one last step by zipping selected_idx and the corresponding predicted classes.
            selected_idx = list(zip(selected_idx, subset_predicted_classes))

        end_al_time = time.time()

        # ===========================
        # RECORD SELECTION STATISTICS
        # ===========================

        # Knowing the disparity between the auto-assigned label and the true label is of interest.
        # We will keep track of this by building auto_assigned_selection_matrix and human_corrected_selection_matrix.
        # *_selection_matrix[i][j] corresponds to how many points of class j were 
        # selected using query set of class i. Ideally, we want high *_selection_matrix[i][i]
        # and low *_selection_matrix[i][j] for i != j.
        
        auto_assigned_selection_matrix = [[0 for y in range(nclasses)] for x in range(nclasses)]
        human_corrected_selection_matrix = [[0 for y in range(nclasses)] for x in range(nclasses)]

        # If there was any part of the budget assigned to the auto-assigned or human-corrected portions, 
        # then record the label counts.

        # Record label counts for auto-assigned points.
        if b3 != 0:
            for i, selected_idx_per_class in enumerate(auto_assigned_selected_idx):
                auto_assigned_selection_matrix[i] = get_label_counts(Subset(full_dataset, [x[0] for x in selected_idx_per_class]), nclasses)

        # Record label counts for human-corrected points.
        if b2 != 0:
            for i, selected_idx_per_class in enumerate(human_corrected_selected_idx):
                human_corrected_selection_matrix[i] = get_label_counts(Subset(full_dataset, [x[0] for x in selected_idx_per_class]), nclasses) 

        # Record selection times.
        auto_selection_time = start_human_time - start_auto_time
        human_selection_time = start_al_time - start_human_time
        al_selection_time = end_al_time - start_al_time

        # Add the recorded information to the experiment dictionary.
        exp_dict['auto_assigned_selected_idx'].append(auto_assigned_selected_idx)
        exp_dict['human_corrected_selected_idx'].append(human_corrected_selected_idx)
        exp_dict['active_learning_selected_idx'].append(selected_idx)
        exp_dict['auto_assigned_selection_matrices'].append(auto_assigned_selection_matrix)
        exp_dict['human_corrected_selection_matrices'].append(human_corrected_selection_matrix)
        exp_dict['auto_assign_selection_times'].append(auto_selection_time)
        exp_dict['human_correct_selection_times'].append(human_selection_time)
        exp_dict['al_selection_times'].append(al_selection_time)

        full_dataset.transform = train_transform # Re-enable any augmentation done during training

        # Update the data to all strategies one last time for this round.
        auto_assign_strategy.update_data(train_dataset, LabeledToUnlabeledDataset(lake_dataset))
        human_correct_strategy.update_data(train_dataset, LabeledToUnlabeledDataset(lake_dataset))
        active_learning_strategy.update_data(train_dataset, LabeledToUnlabeledDataset(lake_dataset))

        print("Auto Selection Time:", auto_selection_time)
        print("Human Selection Time:", human_selection_time)
        print("AL Selection Time:", al_selection_time)
        print('Number of training points -', len(train_dataset))

        # ==============
        # BEGIN TRAINING
        # ==============

        # Start training
        dt.update_data(train_dataset)
        start_train = time.time()
        clf = dt.train(None)
        end_train = time.time()

        # Record the new test accuracy, time spent in training, and corresponding set size
        test_acc = dt.get_acc_on_set(test_dataset)
        exp_dict['test_accuracies'].append(test_acc)
        exp_dict['set_sizes'].append(len(train_dataset))
        exp_dict['train_times'].append(end_train - start_train)

        # Update the model in the strategy
        auto_assign_strategy.update_model(clf)
        human_correct_strategy.update_model(clf)
        active_learning_strategy.update_model(clf)
        print('Testing accuracy:', round(test_acc*100, 2), flush=True)

        # Create a checkpoint
        round_checkpoint = Checkpoint(exp_dict, train_lake_usage_list, assigned_labels, clf.state_dict(), experiment_name=experiment_name)
        round_checkpoint.save_checkpoint(checkpoint_directory)

    print('Training Completed')
    delete_checkpoints(checkpoint_directory, experiment_name)
    return exp_dict

"""### Experiment Fixture Creation"""

class ImageFolderWrapper(Dataset):
    
    def __init__(self, wrapped_dataset, transform=None, target_transform=None):
    
        # Get size info; create data/label arrays
        self.data = []
        self.labels = []
        
        for data_batch, label_batch in wrapped_dataset:
            self.data.append(data_batch)
            self.labels.append(label_batch)
            
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        
        if self.transform is not None:
            data = self.transform(data)
        
        if self.target_transform is not None:
            label = self.transform(label)
            
        return data, label
    
    def __len__(self):
        return len(self.labels)

def ensure_three_channel(image_tensor):
    
    if image_tensor.shape[0] == 1:  # If monochrome
        return torch.repeat_interleave(image_tensor, 3, dim=0)
    elif image_tensor.shape[0] == 4:  # If alpha layer
        return image_tensor[:3,:,:]
    else:
        return image_tensor

class CarsDataset(Dataset):
    """
    PyTorch interface for Stanford Cars-196 dataset. Thanks to https://github.com/dtiarks/pytorch_cars
    for the skeleton.
    """

    def __init__(self, root_directory, train=True, download=False, transform=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        cars_root = os.path.join(root_directory, "cars196")
        cars_meta = os.path.join(cars_root, "devkit", "cars_meta.mat")

        # Download the dataset if needed.
        if download:
            archives = [("http://ai.stanford.edu/~jkrause/car196/cars_train.tgz", "cars_train.tgz"),
                        ("http://ai.stanford.edu/~jkrause/car196/cars_test.tgz", "cars_test.tgz"),
                        ("https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz", "car_devkit.tgz")]
            for archive, file_to_check in archives:
                check_path = os.path.join(cars_root, file_to_check)
                if os.path.exists(check_path):
                    continue
                
                download_and_extract_archive(archive, cars_root)

        # Set annotations + data source depending on split
        if train:
            mat_anno = os.path.join(cars_root, "devkit", "cars_train_annos.mat")
            self.data_dir = os.path.join(cars_root, "cars_train")
        else:
            mat_anno = os.path.join(cars_root, "devkit", "cars_test_annos.mat")
            self.data_dir = os.path.join(cars_root, "cars_test")

        self.full_data_set = loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations'][0]
        self.car_names = np.array(loadmat(cars_meta)['class_names'][0])

        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name)
        car_class = self.car_annotations[idx][-2][0][0] - 1 # zero-index the class

        if self.transform:
            image = self.transform(image)

        return image, car_class
    
class BirdsDataset(Dataset):

    def __init__(self, root_directory, train=True, download=False, transform=None):

        birds_root = os.path.join(root_directory, "caltech_birds")

        # Download if needed
        if download:
            archives = [("http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz", "CUB_200_2011.tgz")]
            for archive, file_to_check in archives:
                check_path = os.path.join(birds_root, file_to_check)
                if os.path.exists(check_path):
                    continue
                
                download_and_extract_archive(archive, birds_root)

        # Get ID-to-filename map
        id_to_filename_map = {}
        id_to_image_path = os.path.join(birds_root, "CUB_200_2011", "images.txt")
        with open(id_to_image_path, "r") as map_file:
            map_reader = csv.reader(map_file, delimiter = " ")
            for (image_id, rel_path) in map_reader:
                id_to_filename_map[int(image_id)] = rel_path

        # Get ID-to-class map
        id_to_class_map = {}
        id_to_class_path = os.path.join(birds_root, "CUB_200_2011", "image_class_labels.txt")
        with open(id_to_class_path, "r") as map_file:
            map_reader = csv.reader(map_file, delimiter = " ")
            for (image_id, class_label) in map_reader:
                id_to_class_map[int(image_id)] = int(class_label) - 1  # Subtract 1 for pytorch labeling scheme.                

        # Get the train-test split
        train_test_split_path = os.path.join(birds_root, "CUB_200_2011", "train_test_split.txt")
        split_subset = []
        with open(train_test_split_path, "r") as split_file:
            split_reader = csv.reader(split_file, delimiter = " ")
            for image_id, is_train_image in split_reader:
                if int(is_train_image) and train:
                    split_subset.append(int(image_id))
                elif not int(is_train_image) and not train:
                    split_subset.append(int(image_id))

        # Get list of filepaths and corresponding classes
        self.filepaths = []
        self.classes = []
        image_folder_root = os.path.join(birds_root, "CUB_200_2011", "images")
        for image_id in split_subset:
            self.filepaths.append(os.path.join(image_folder_root, id_to_filename_map[image_id]))
            self.classes.append(id_to_class_map[image_id])

        self.transform = transform

    def __getitem__(self, index):

        img_name = self.filepaths[index]
        label = self.classes[index]
        image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.classes)

class DogsDataset(Dataset):

    def __init__(self, root_directory, train=True, download=False, transform=None):

        dogs_root = os.path.join(root_directory, "stanford_dogs")

        # Download if needed
        if download:
            archives = [("http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar","images.tar"),
                        ("http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar","annotation.tar"),
                        ("http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar","lists.tar")]
            for archive, file_to_check in archives:
                check_path = os.path.join(dogs_root, file_to_check)
                if os.path.exists(check_path):
                    continue
                
                download_and_extract_archive(archive, dogs_root)

        if train:
            dataset_mat_path = os.path.join(dogs_root, "train_list.mat")
        else:
            dataset_mat_path = os.path.join(dogs_root, "test_list.mat")

        dataset_mat = loadmat(dataset_mat_path)

        self.filepaths = []
        for file_name in dataset_mat['file_list']:
            file_name = file_name[0][0]
            filepath = os.path.join(dogs_root, "Images", file_name)
            self.filepaths.append(filepath)

        self.labels = []
        for label in dataset_mat['labels']:
            label = label[0] - 1
            self.labels.append(label)

        self.transform = transform

    def __getitem__(self, index):

        img_name = self.filepaths[index]
        label = self.labels[index]
        image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)

def get_tiny_imagenet(dataset_root_path):

    # Download and extract TinyImageNet if it isn't already.
    filepath = os.path.join(dataset_root_path, "tiny-imagenet-200.zip")
    if not os.path.exists(filepath):
        download_command = F"wget -P {dataset_root_path} http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        os.system(download_command)

    dataset_path = os.path.join(dataset_root_path, "tiny-imagenet-200")
    if not os.path.exists(dataset_path):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dataset_root_path)

    # TinyImageNet has a test set, but it's labels are not available (following good practice).
    # Hence, we must evaluate on the validation set. We prepare the validation set according to 
    # https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f
    # so that PyTorch's ImageFolder class can be used.
    validation_dir = os.path.join(dataset_path, 'val')

    # Open and read val annotations text file
    with open(os.path.join(validation_dir, 'val_annotations.txt'), 'r') as fp:
        data = fp.readlines()

    # Create image filename to class dictionary
    val_image_filename_to_class_dict = {}
    for line in data:
        words = line.split('\t')
        val_image_filename_to_class_dict[words[0]] = words[1]

    # Map each image into its own class folder
    old_val_img_dir = os.path.join(validation_dir, 'images')
    for img, folder in val_image_filename_to_class_dict.items():
        newpath = (os.path.join(validation_dir, folder, 'images'))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(old_val_img_dir, img)):
            os.rename(os.path.join(old_val_img_dir, img), os.path.join(newpath, img))
    if os.path.exists(old_val_img_dir):
        os.rmdir(old_val_img_dir)
        

def get_experiment_fixture(dataset_root_path, dataset_name, seed_set_size, model_name, model_base_path, init_model_train_args):

    # Load the dataset
    if dataset_name == "CIFAR10":

        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        full_train_dataset = datasets.CIFAR10(dataset_root_path, download=True, train=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(dataset_root_path, download=True, train=False, transform=test_transform)

        nclasses = 10 # NUM CLASSES HERE

    elif dataset_name == "CIFAR100":

        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        full_train_dataset = datasets.CIFAR100(dataset_root_path, download=True, train=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(dataset_root_path, download=True, train=False, transform=test_transform)

        nclasses = 100 # NUM CLASSES HERE

    elif dataset_name == "MNIST":

        image_dim=28
        train_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        full_train_dataset = datasets.MNIST(dataset_root_path, download=True, train=True, transform=train_transform)
        test_dataset = datasets.MNIST(dataset_root_path, download=True, train=False, transform=test_transform)

        nclasses = 10 # NUM CLASSES HERE

    elif dataset_name == "TinyImageNet":

        get_tiny_imagenet(dataset_root_path)

        train_transform = transforms.Compose([transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std

        # Use val as test
        train_path = os.path.join(dataset_root_path, "tiny-imagenet-200", "train")
        test_path = os.path.join(dataset_root_path, "tiny-imagenet-200", "val")
        
        # Load TinyImageNet into memory instead of resorting to ImageFolder, which is slow.
        full_train_dataset = datasets.ImageFolder(train_path)
        test_dataset = datasets.ImageFolder(test_path)
        full_train_dataset = ImageFolderWrapper(full_train_dataset, transform=train_transform)
        test_dataset = ImageFolderWrapper(test_dataset, transform=test_transform)  

        nclasses = 200 
    
    elif dataset_name == "SVHN":
        
        image_dim = 32
        
        train_transform = transforms.Compose([transforms.RandomCrop(image_dim, padding=4), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std

        full_train_dataset = datasets.SVHN(dataset_root_path, split='train', download=True, transform=train_transform, target_transform=torch.tensor)
        test_dataset = datasets.SVHN(dataset_root_path, split='test', download=True, transform=test_transform, target_transform=torch.tensor)
        
        nclasses = 10 # NUM CLASSES HERE
        
    elif dataset_name == "Birds":
        
        pre_crop_size = 256
        image_dim = 224
        train_transform = transforms.Compose([
                                transforms.Resize((pre_crop_size, pre_crop_size)),
                                transforms.RandomCrop(image_dim),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ensure_three_channel,
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
        test_transform = transforms.Compose([
                                transforms.Resize((pre_crop_size, pre_crop_size)),
                                transforms.CenterCrop(image_dim),
                                transforms.ToTensor(),
                                ensure_three_channel,
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
        
        full_train_dataset = BirdsDataset(dataset_root_path, download=True, train=True, transform=train_transform)
        test_dataset = BirdsDataset(dataset_root_path, download=True, train=False, transform=test_transform)
        
        nclasses = 200
        
    elif dataset_name == "Cars":
        
        pre_crop_size = 256
        image_dim = 224
        train_transform = transforms.Compose([
                                transforms.Resize((pre_crop_size, pre_crop_size)),
                                transforms.RandomCrop(image_dim),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ensure_three_channel,
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
        test_transform = transforms.Compose([
                                transforms.Resize((pre_crop_size, pre_crop_size)),
                                transforms.CenterCrop(image_dim),
                                transforms.ToTensor(),
                                ensure_three_channel,
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
        
        full_train_dataset = CarsDataset(dataset_root_path, train=True, download=True, transform=train_transform)
        test_dataset = CarsDataset(dataset_root_path, train=False, download=True, transform=test_transform)

        nclasses = 196
        
    elif dataset_name == "Dogs":
        
        pre_crop_size = 256
        image_dim = 224
        train_transform = transforms.Compose([
                                transforms.Resize((pre_crop_size, pre_crop_size)),
                                transforms.RandomCrop(image_dim),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ensure_three_channel,
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
        test_transform = transforms.Compose([
                                transforms.Resize((pre_crop_size, pre_crop_size)),
                                transforms.CenterCrop(image_dim),
                                transforms.ToTensor(),
                                ensure_three_channel,
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
        
        full_train_dataset = DogsDataset(dataset_root_path, train=True, download=True, transform=train_transform)
        test_dataset = DogsDataset(dataset_root_path, train=False, download=True, transform=test_transform)

        nclasses = 120


    if model_name == "resnet18":
        model = ResNet18(num_classes=nclasses)
    elif model_name == "vresnet18":
        model = VariableSizeResNet18(pretrained=True)
        last_layer_features = model.fc.in_features
        model.fc = nn.Linear(last_layer_features, nclasses)
    elif model_name == "mnistnet":
        model = MnistNet()
    else:
        raise ValueError("Add model implementation")

    # Seed the rng used in dataset splits
    np.random.seed(42)

    # Retrieve the labels of the training set
    train_labels = get_labels(full_train_dataset)

    # Derive a list of indices that will represent the training set indices. The rest will represent the unlabeled set indices.
    per_class_size = seed_set_size // nclasses
    initial_train_idx = []
    for cls in range(nclasses):

        # Sample random points per class to form a balanced seed
        cls_idx = torch.where(train_labels==cls)[0]
        chosen_idx = np.random.choice(cls_idx, size=per_class_size, replace=False)
        initial_train_idx.extend(chosen_idx)

    # See if a model has already been trained for this fixture.
    model_name = F"{dataset_name}_{model_name}_{seed_set_size}"
    model_save_path = os.path.join(model_base_path, model_name)

    if os.path.isfile(model_save_path):
        print("Found Initial Model")
        state_dict = torch.load(model_save_path)
        model.load_state_dict(state_dict)
    else:
        print("Training Initial Model...")
        init_trainer = data_train(Subset(full_train_dataset, initial_train_idx), model, init_model_train_args)
        model = init_trainer.train(None)
        torch.save(model.state_dict(), model_save_path)

    return full_train_dataset, test_dataset, initial_train_idx, model, nclasses

if __name__ == "__main__":

    
    parser = ArgumentParser()
    
    # Arguments for script
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--al_strategy", type=str, required=True, help="AL strategy to use")
    parser.add_argument("--human_correct_strategy", type=str, required=True, help="Human correction strategy to use")
    parser.add_argument("--auto_assign_strategy", type=str, required=True, help="Auto-assign strategy to use")
    parser.add_argument("--b1", type=int, required=True, help="Pure AL budget")
    parser.add_argument("--b2", type=int, required=True, help="Human correction budget")
    parser.add_argument("--b3", type=int, required=True, help="Auto-assign budget")
    parser.add_argument("--seed_size", type=int, required=True, help="Initial size of labeled set")
    parser.add_argument("--rounds", type=int, required=True, help="Number of selection rounds")
    parser.add_argument("--runs", default=1, type=int, required=True, help="Number of experiment trials")
    parser.add_argument("--device", default=0, type=int, help="CUDA Device ID")
    parser.add_argument("--thread_count", default=3, type=int, help="Num. Threads for SMI")
    parser.add_argument("--num_partitions_human", default=5, type=int, help="Lake set partitions for human-corrected strategy")
    parser.add_argument("--adaptive", default=False, type=bool, help="Vary suggested budget linearly from full HC to provided b2,b3 values over the course of the provided rounds")
    
    # Parse the arguments
    init_args = parser.parse_args()
    
    dataset = init_args.dataset
    al_strategy = init_args.al_strategy
    human_correct_strategy = init_args.human_correct_strategy
    auto_assign_strategy = init_args.auto_assign_strategy
    b1 = init_args.b1
    b2 = init_args.b2
    b3 = init_args.b3
    seed_size = init_args.seed_size
    rounds = init_args.rounds
    per_exp_runs = init_args.runs
    device = init_args.device
    thread_count = init_args.thread_count
    adaptive = init_args.adaptive

    # Define training arguments
    args = {'islogs': False,
                'optimizer': 'sgd',
                'isverbose': True,
                'isreset': True,
                'max_accuracy': 0.99,
                'n_epoch': 400,
                'lr': 0.001,
                'device': 'cuda:'+ str(device),
                'batch_size': 64,
                'thread_count': thread_count,
                'metric': 'cosine',
                'embedding_type': 'gradients',
                'gradType': 'bias_linear'}  

    # Decide which models to use, depending on the dataset
    if dataset == 'cifar10':
        dataset_name = "CIFAR10"
        model_name = "resnet18"
                  
    elif dataset == 'mnist':
        dataset_name = "MNIST"
        model_name = "mnistnet"

    elif dataset == 'cifar100':
        dataset_name = "CIFAR100"
        model_name = "resnet18"
    elif dataset == "svhn":
        dataset_name = "SVHN"
        model_name = "resnet18"
    elif dataset == "tinyimagenet":
        dataset_name = "TinyImageNet"
        model_name = "vresnet18"
    elif dataset == "birds":
        dataset_name = "Birds"
        model_name = "vresnet18"
        args['pretrained_class_count'] = 200
        args['should_freeze'] = True
        args['max_accuracy'] = 0.95
    elif dataset == "cars":
        dataset_name = "Cars"
        model_name = "vresnet18"
        args['pretrained_class_count'] = 196
        args['should_freeze'] = True
        args['max_accuracy'] = 0.95
    elif dataset == "dogs":
        dataset_name = "Dogs"
        model_name = "vresnet18"
        args['pretrained_class_count'] = 120
        args['should_freeze'] = True
        args['max_accuracy'] = 0.95
    else:
        raise ValueError("experiment_name must be one of 'cifar10', 'cifar100', 'mnist', 'svhn', 'tinyimagenet', 'birds', 'cars', 'dogs'")

    # Create saving directories
    mount_point_directory = "."

    google_drive_directory = "results/" + dataset
    base_save_directory = os.path.join(mount_point_directory, google_drive_directory)

    google_drive_directory = "check/"
    checkpoint_directory = os.path.join(mount_point_directory, google_drive_directory)

    google_drive_directory = "model/"
    model_directory = os.path.join(mount_point_directory, google_drive_directory)

    dataset_root_directory = os.path.join(mount_point_directory, "datasets")

    os.makedirs(checkpoint_directory, exist_ok=True)
    os.makedirs(model_directory, exist_ok=True)
    os.makedirs(dataset_root_directory, exist_ok=True)

    # Set new proc. start method to spawn to allow for CUDA usage
    mp.set_start_method("spawn", force=True)

    # Given the auto-assign / human-correct / al strategy, different partition sizes might be needed.
    # They are assigned here.
    args['num_partitions_al'] = 1
    args['num_partitions_auto'] = 1
    args['num_partitions_human'] = init_args.num_partitions_human
    
    if dataset == "cifar100" or dataset == "tinyimagenet":
        args['num_partitions_auto'] = 3
        if al_strategy == "badge":
            args['num_partitions_al'] = 5

    # Ensure that the result location exists
    auto_label_results_save_directory = os.path.join(base_save_directory, F"{al_strategy}", F"{seed_size}")
    os.makedirs(auto_label_results_save_directory, exist_ok=True)

    # Obtain experiment fixture.
    train_dataset, test_dataset, init_train_idx, model, nclasses = get_experiment_fixture(dataset_root_directory, dataset_name, seed_size, model_name, model_directory, args)
    init_train_lake_usage_list = [1 if i in init_train_idx else 0 for i in range(len(train_dataset))]

    # Repeat experiment for prescribed number of runs
    for run_count in range(per_exp_runs):

        # Get the results file name under which the results should be saved    
        experiment_results_file_name = F"{dataset}_{al_strategy}_{human_correct_strategy}_{auto_assign_strategy}_{b1}_{b2}_{b3}_{seed_size}_{rounds}_{run_count}_{adaptive}.json"
        experiment_results_path = os.path.join(auto_label_results_save_directory, experiment_results_file_name)

        # Determine if this experiment needs to be run
        if not os.path.isfile(experiment_results_path):
            print("======================================")
            print(F"Running {experiment_results_file_name}")
            print("======================================")

            # There is no data for this run. Run this experiment again.
            results = al_train_loop(train_dataset, copy.deepcopy(init_train_lake_usage_list), test_dataset, copy.deepcopy(model), rounds, b1, b2, b3, args, nclasses, al_strategy, human_correct_strategy, auto_assign_strategy, checkpoint_directory, experiment_results_file_name, adaptive)
            with open(experiment_results_path, "w") as write_file:
                json.dump(results, write_file)
        else:
            print("======================================")
            print(F"Results already obtained; skipping {experiment_results_file_name}")
            print("======================================")
