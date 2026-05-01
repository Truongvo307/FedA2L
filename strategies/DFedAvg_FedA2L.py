import copy
import logging
import os
import time
from argparse import Namespace
import math
import numpy as np
import re
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import polars as pl
import ast
import json

from yaml import parser
from .base import Coordinator, Node
import pandas as pd
from collections import defaultdict 
optional = {"max_lr_upper_bound": 0.02, 
            "max_lr_lower_bound": 0.0001, 
            "swin":10, 
            "time_tunning":1,
            "metric_tunning":"[wds,crs]",
            "start_tunning":10,
            "kt":0.1,
            "ratio_tunning":[0.6,0.4]
            }

# Argument parser update function
def args_update(parser):
    parser.add_argument("--max_lr_upper_bound", type=float, default=None)
    parser.add_argument("--max_lr_lower_bound", type=float, default=None)
    parser.add_argument("--swin", type=int, default=None)
    parser.add_argument("--start_tunning", type=int, default=None)
    parser.add_argument("--time_tunning", type=int, default=None)
    parser.add_argument("--metric_tunning", type=str, default=None)
    parser.add_argument("--kt", type=float, default=None)
    parser.add_argument("--ratio_tunning", type=float, nargs='+', default=[0.6, 0.4])

class DFedAvg_FedA2L(Coordinator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_class = DFedAvg_FedA2L_Node
        self.total_times = args[0].times

    def find_first_occurrence(self,df, threshold):
        result = df[df["mean_acc"] >= threshold].index.min()
        return result if pd.notna(result) else None


class DFedAvg_FedA2L_Node(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculate_metric = ModelMetrics(self.metrics)
        self.count = 0
        self.metric_t = setattr(self, f"{self.metric_tunning}_t", {})
        self.metric_a = setattr(self, f"{self.metric_tunning}_a", {})
        self.metric_tunning = self.metric_tunning.strip("[]").split(",")

    def train(self):
        self.betrain_model = copy.deepcopy(self.model)
        super().train()
        self.metric_t = self.calculate_metric.calculate_metric(self.metric_tunning[0],self.betrain_model,self.model)

    def aggregation(self):
        beagg_model = copy.deepcopy(self.model) 
        super().aggregation()
        self.metric_a = self.calculate_metric.calculate_metric(self.metric_tunning[1],beagg_model,self.model)
        if self.iteration > self.start_tunning and self.iteration % self.time_tunning==0:
            self.learning_rate_tunning()
            self.count +=1
        else:   
            for group in self.optimizer.param_groups:
                self.metrics.setdefault(f"{group['name']}_lr", []).append(group['lr'])
                print(f"Node {self.id} | {group['name']}_lr: {group['lr']}")
        self.calculate_metric.update_metrics(self.metric_tunning,self.metric_t,self.metric_a)
    

    def learning_rate_tunning(self): 
        layerwise_lr = {}
        epsilon = 1e-6
        for layer,group in zip(self.metric_t.keys(),self.optimizer.param_groups):  
            metrics_keys = [f"{layer}_{metric}_{state}" for metric, state in zip(self.metric_tunning, ["t", "a"])]
            metrics_swin = [self.metrics[key][-self.swin:] if len(self.metrics[key]) >= self.swin else self.metrics[key] for key in metrics_keys]
            metrics_tensor = torch.tensor(metrics_swin, dtype=torch.float32)
            
            metric_mean = torch.mean(metrics_tensor, dim=1)
            metric_std = torch.std(metrics_tensor, dim=1) + epsilon
            
            metric_score = (metrics_tensor[:, -1] - metric_mean) / metric_std
            lambda_l = torch.dot(torch.tensor(self.ratio_tunning, dtype=torch.float32), torch.exp(metric_score))
            layer_lr = self.learning_rate * (1 + torch.tanh(torch.log(lambda_l)))
            efr = 1 / torch.sqrt(torch.tensor(1 + self.kt * self.count, dtype=torch.float32))
            update_layer_lr = torch.clamp(layer_lr* efr, self.max_lr_lower_bound, self.max_lr_upper_bound) 
            lr  = update_layer_lr
            layerwise_lr[layer] = group["lr"] = lr
            self.metrics.setdefault(f"{group['name']}_lr", []).append(update_layer_lr.item())


    def get_optimizer(self):
        self.layer_names = set()
        for name, param in self.model.named_parameters():
            if name.split(".")[-1].startswith(("weight", "bias")):
                layer_name = ".".join(name.split(".")[:-1])
                self.layer_names.add(layer_name)
            
        self.optimizer_class = getattr(__import__("optimizers"), self.optimizer)
        optimizer_params = {"lr": self.learning_rate}
        
        
        param_groups = []
        module_param_dict = defaultdict(list)
        assigned_param_ids = set()

        for param_name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue 

            param_id = id(param)
            if param_id in assigned_param_ids:
                print(f"Warning: Parameter {param_name} (id={param_id}) appears shared/aliased. Skipping.")
                continue

            parts = param_name.split('.')
            if len(parts) == 1: 
                module_name = "" 
            else:
                module_name = ".".join(parts[:-1])

            module_param_dict[module_name].append(param)
            assigned_param_ids.add(param_id) 

        cnt = 0
        final_assigned_param_ids = set() 

        for module_name, params_list in module_param_dict.items():
            group_name = module_name if module_name else "model_top_level"
            if not params_list: 
                continue
            if len(self.layer_names) > 10:
                if 'bn' in module_name:
                    current_lr = self.learning_rate*self.kt*0.5
                else: 
                    current_lr = self.learning_rate*self.kt
            else:  
                current_lr = self.learning_rate
                if self.optimizer.lower() == "sgd" and hasattr(self, "momentum"):
                    optimizer_params["momentum"] = self.momentum
            param_groups.append({
                "params": params_list,     
                "lr": current_lr,
                "name": group_name, 
            })

            for p in params_list:
                final_assigned_param_ids.add(id(p))
            cnt += 1   

        self.optimizer = self.optimizer_class(param_groups, **optimizer_params)




class ModelMetrics:
    def __init__(self,metrics):
        self.metrics = metrics
        self.methods = ["wds", "crs", "css", "wvs"]

    def softsign(self, x):
        return x / (1 + np.abs(x))
    
    def calculate_divergence(self, model_before, model_after): 
        divergence_dict = {}
        for (name, param_before), (_, param_after) in zip(model_before.named_parameters(), model_after.named_parameters()):
            param_diff = param_after - param_before
            divergence = (torch.norm(param_diff) / torch.norm(param_before + 1e-8)).cpu().detach().numpy()
            divergence_dict[name] = self.softsign(divergence.item())
        return self.remove_bias(divergence_dict)

    def calculate_consensus_ratio(self, model_before, model_after, threshold=1e-3):
        consensus_dict = {}
        for (name, param_before), (_, param_after) in zip(model_before.named_parameters(), model_after.named_parameters()):
            param_diff = param_after - param_before
            consensus = torch.sum(torch.abs(param_diff) < threshold).item() / param_diff.numel()
            consensus_dict[name] = self.softsign(consensus)
        return self.remove_bias(consensus_dict)

    def calculate_all_metrics(self, model_before, model_after):
        """
        Compute all metrics (Weight Divergence, Consensus Ratio, Convergence Speed, Weight Variance)
        """
        all_metrics = {
            "wds": self.calculate_divergence(model_before, model_after),
            "crs": self.calculate_consensus_ratio(model_before, model_after),
        }
        self.save_all_metrics(all_metrics)
        return 
    
    
    def update_metrics(self,metric_tunning,metricts_t,metricts_a):
        for (layer, metrict_t), metrict_a  in zip(
            metricts_t.items(), metricts_a.values()
        ):
            self.metrics.setdefault(f"{layer}_{metric_tunning[0]}_t", []).append(metrict_t)
            self.metrics.setdefault(f"{layer}_{metric_tunning[1]}_a", []).append(metrict_a)

    def calculate_metric(self, method_type, model_before, model_after):

        method_map = {
            "wds": self.calculate_divergence,
            "crs": self.calculate_consensus_ratio,
        }

        if method_type not in method_map:
            raise ValueError(f"Invalid method type: {method_type}. Choose from {list(method_map.keys())}")

        return method_map[method_type](model_before, model_after)

    def save_all_metrics(self, metrics_dict, name):
        for layer in metrics_dict["wds"].keys():
            for key in self.methods:
                self.metrics.setdefault(f"{layer}_{key}_{name}", []).append(metrics_dict[key][layer])

    def remove_bias (self,metrics):
        return {k.rsplit('.weight', 1)[0]: v for k, v in metrics.items() if '.weight' in k}
    
    def get_metrics(self):
        return self.metrics
