import copy
import logging
import os
import re
import sys
import time
from argparse import Namespace
from collections import deque
import ray
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml 


class SharedMethods:

    @staticmethod
    def zero_parameters(model):
        zero = copy.deepcopy(model)
        for param in zero.parameters():
            param.data.zero_()
        return zero

    @staticmethod
    def update_model_params(old, new):
        """Update the parameters of old_model with those from new_model."""
        for old_param, new_param in zip(old.parameters(), new.parameters()):
            old_param.data.copy_(new_param.data)

    @staticmethod
    def update_optimizer_params(old, new):
        """Update the parameters and hyperparameters of old_optimizer with those from new_optimizer."""
        for old_group, new_group in zip(old.param_groups, new.param_groups):
            for key in new_group.keys():
                if key != "params": 
                    old_group[key] = new_group[key]

            # Update the model parameters inside param_groups
            for old_param, new_param in zip(old_group["params"], new_group["params"]):
                old_param.data.copy_(new_param.data)


    def make_logger(self, name, path):
        log_path = os.path.join(path, f"{name.lower().strip()}.log")

        logger_name = f"{name}_{self.times}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_path)
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            f"%(asctime)s ~ %(levelname)s ~ %(lineno)-4.4d ~ {name} ~ %(message)s"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info(f"Logger created at {log_path}")

    def save_results(self):
        pl_df = pl.DataFrame(self.metrics)
        path = os.path.join(self.result_path, self.name.lower().strip() + ".csv")
        pl_df.write_csv(path)
        self.logger.info(f"Results saved to {path}")

    def mkdir(self):
        self.save_path = os.path.join(self.save_path, str(self.times))
        self.model_path = os.path.join(self.save_path, "models")
        self.model_info_path = os.path.join(self.save_path, "models_info")
        self.log_path = os.path.join(self.save_path, "logs")
        self.result_path = os.path.join(self.save_path, "results")
        for dir in [
            self.save_path,
            self.model_path,
            self.log_path,
            self.model_info_path,
            self.result_path,
        ]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def set_configs(self, configs, **kwargs):
        if isinstance(configs, Namespace):
            for key, value in vars(configs).items():
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.configs = configs

    @staticmethod
    def get_size(obj):
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement() / (1024**2)  # Size in MB
        if isinstance(obj, torch.nn.Module):
            total_size = sum(
                param.element_size() * param.nelement() for param in obj.parameters()
            )
            total_size += sum(
                buffer.element_size() * buffer.nelement() for buffer in obj.buffers()
            )  # Include buffers
            return total_size / (1024**2)  # Size in MB
        if isinstance(obj, DataLoader):
            total_size = sum(
                sum(
                    item.element_size() * item.nelement()
                    for item in data
                    if isinstance(item, torch.Tensor)
                )
                for data in obj.dataset
            )
            return total_size / (1024**2)  # Size in MB
        return sys.getsizeof(obj) / (1024**2)  # Size in MB


class Coordinator(SharedMethods):
    def __init__(self, configs, t):
        self.set_configs(configs=configs, times=t)
        self.mkdir()
        self.name = "COORDINATOR"
        self.make_logger(name=self.name, path=self.log_path)
        self.metrics = {"mean_acc": [], "mean_loss": [], "time_used": []}
        self.node_class = Node
        if self.parallel:
            self.devices = [f"{self.device_id}:{i}" for i in self.device_id.split(",")]
            self.num_gpus = len(self.devices)
            ray.init(
                num_gpus=self.num_gpus,
                ignore_reinit_error=True,
                logging_level=logging.ERROR,
                log_to_driver=False,
            )

    def create_node(self):
        self.nodes = []
        
        for i in range(self.num_nodes):
            node_configs = copy.deepcopy(self.configs)
            self.nodes.append(
                self.node_class(
                    configs=self.configs,
                    id=i,
                    times=self.times,
                    neighbor_id=self.topology[i],
                )
            )

    def get_topology(self):
        params = {"num_nodes": self.num_nodes}
        if self.topology == "KConnected":
            params["k"] = self.k

        self.topology = getattr(__import__("topologies"), self.topology)(
            **params
        ).neighbors

    def _train(self, i):
        for node in self.nodes:
            node.iteration = i
            node.train()

    def _get_neighbors_data(self):
        for node in self.nodes:
            node.uploaded_weights = [node.train_samples]
            node.uploaded_models = [copy.deepcopy(node.model)]

            b = 0
            b += self.get_size(node.uploaded_weights[-1])
            b += self.get_size(node.uploaded_models[-1])
            node.metrics["send_mb"].append(b * len(node.neighbor_id))

            b = 0
            for id in node.neighbor_id:
                node.uploaded_weights.append(self.nodes[id].train_samples)
                b += self.get_size(node.uploaded_weights[-1])
                node.uploaded_models.append(copy.deepcopy(self.nodes[id].model))
                b += self.get_size(node.uploaded_models[-1])
            node.metrics["receive_mb"].append(b)

    def _aggregation(self):
        for node in self.nodes:
            node.aggregation()

    def evaluate(self):
        acc_nodes, loss_nodes, tloss_nodes = [], [], []
        for node in self.nodes:
            acc, loss = node.test_metrics()
            acc_nodes.append(acc)
            loss_nodes.append(loss)
            tloss = node.train_metrics()
            tloss_nodes.append(tloss)
            node.logger.info(
                f"Test Accurancy: {acc*100:06.3f}% | Test Loss: {loss:06.3f} | Train Loss: {tloss:06.3f}"
            )
        self.logger.info(
            f"Test Accurancy: {np.mean(acc_nodes)*100:06.3f}% | Test Loss: {np.mean(loss_nodes):06.3f} | Train Loss: {np.mean(tloss_nodes):06.3f}"
        )
        self.metrics["mean_acc"].append(np.mean(acc_nodes))
        self.metrics["mean_loss"].append(np.mean(loss_nodes))

    def _post_train(self):
        pass

    def save_results(self):
        super().save_results()
        for node in self.nodes:
            node.save_results()

    def save_validate_files(self):
        dfs = []
        mean_acc = []
        mean_loss = []
        time_used = []
        mean_df = []
        sw = 10
        for i in range(self.times):
            file_name = os.path.join(
                self.save_path[:-1],
                str(i),
                "results",
                self.name.lower().strip() + ".csv",
            )
            df = pl.read_csv(file_name)
            mean_acc.append(df["mean_acc"].to_list())
            mean_loss.append(df["mean_loss"].to_list())
            time_used.append(df["time_used"].to_list())
            dfs.append(df)
        
        mean_df = sum(dfs) / len(dfs)
        mean_df.write_csv(os.path.join(self.save_path[:-1], "mean_data.csv"))
        print(f"Results saved to: {os.path.join(self.save_path[:-1], 'mean_data.csv')}")
        rollingdf = mean_df.select([
        pl.col(col).rolling_mean(window_size=sw).alias(col)
        for col in mean_df.columns
        ])
        rollingdf.write_csv(os.path.join(self.save_path[:-1], f'mean_data_rolling_sw{sw}.csv'))
        convergence = []
        convergence_sw = []
        for target in self.convergence_targets:
            idx = (mean_df["mean_acc"] >= target / 100).arg_max()
            convergence.append({"target": target, "first_index": idx})
            idx_sw = (rollingdf["mean_acc"] >= target/100).arg_max()
            convergence_sw.append({"target": target, "first_index": idx_sw})    
        convergence_df = pl.DataFrame(convergence)
        convergence_df.write_csv(
            os.path.join(self.save_path[:-1], "convergence_analysis.csv")
        )
        print(
            f"Results saved to: {os.path.join(self.save_path[:-1],'convergence_analysis.csv')}"
        )
        convergence_sw_df = pl.DataFrame(convergence_sw)
        convergence_sw_df.write_csv(os.path.join(self.save_path[:-1], 'convergence_analysis_sw.csv'))
        print(f"Results saved to: {os.path.join(self.save_path[:-1], 'convergence_analysis_sw.csv')}")

    def run(self):
        self.get_topology()
        self.create_node()
        for i in range(self.iterations):
            s = time.time()
            self.logger.info("=" * 23 + f" Iteration: {str(i).zfill(4)} " + "=" * 24)
            self._train(i)
            self._post_train()
            self._get_neighbors_data()
            self._aggregation()
            self.evaluate()
            self.metrics["time_used"].append(time.time() - s)
        self.save_results()
        if self.convergence_targets and self.times == self.configs.times - 1:
            self.convergence_targets = [
                float(x) for x in re.findall(r"\d+", self.convergence_targets)
            ]
            self.save_validate_files()


class Node(SharedMethods):
    default_value = -1.0

    def __init__(self, configs: dict, id: int, times, neighbor_id: list):
        self.set_configs(configs=configs, id=id, times=times, neighbor_id=neighbor_id)
        self.mkdir()
        self.name = f"NODEID_{str(self.id).zfill(4)}"
        self.make_logger(name=self.name, path=self.log_path)
        self.logger.info(f"Neighbor: {self.neighbor_id}")

        dataset_path = self.configs.dataset_path
        self.train_file = os.path.join(dataset_path, "train/", str(self.id) + ".npz")
        f = f"{self.id}.npz" if self.configs.eval_method == "personalization" else "server.npz"
        self.test_file = os.path.join(dataset_path, "test", f)

        config_path = os.path.join(dataset_path, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            if 'num_classes' in dataset_config:
                self.configs.num_classes = dataset_config['num_classes']
                self.logger.info(f"Loaded true num_classes from config: {self.configs.num_classes}")
            if 'vocab_size' in dataset_config:
                self.configs.vocab_size = dataset_config['vocab_size']
                self.logger.info(f"Loaded true vocab_size from config: {self.configs.vocab_size}")
        else:
             self.logger.warning(f"config.yaml not found. Using default parameters.")

        self.get_model()
        
        self.metrics = {
            "train_time": [], "accs": [], "losses": [], "losses_test": [],
            "send_mb": [], "receive_mb": [],
        }
        self.get_loss()
        self.get_optimizer()

    def mkdir(self):
        self.save_path = os.path.join(self.save_path, str(self.times))
        self.model_path = os.path.join(self.save_path, "models")
        self.model_info_path = os.path.join(self.save_path, "models_info")
        self.log_path = os.path.join(self.save_path, "logs")
        self.result_path = os.path.join(self.save_path, "results")
        for dir in [
            self.save_path,
            self.model_path,
            self.log_path,
            self.model_info_path,
            self.result_path,
        ]:
            if not os.path.exists(dir):
                os.makedirs(dir)


    def save_model(self):
        path = os.path.join(self.configs.model_path, self.name.lower().strip() + ".pt")
        torch.save(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def load_data(self, path):
        with open(path, "rb") as f:
            data = np.load(f, allow_pickle=True)["data"].tolist()
        if self.configs.dataset.lower() in ['sent140', 'shakespeare']:
            x = torch.tensor(data["x"], dtype=torch.long)
        else:
            x = torch.tensor(data["x"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.long)
        return torch.utils.data.TensorDataset(x, y)

    def get_loss(self):
        self.loss = getattr(__import__("losses"), self.configs.loss)()

    def get_optimizer(self):
        optimizer_class = getattr(__import__("optimizers"), self.configs.optimizer)
        optimizer_params = {"lr": self.configs.learning_rate}
        if self.configs.optimizer.lower() == "sgd" and hasattr(self.configs, "momentum"):
            optimizer_params["momentum"] = self.configs.momentum
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

    def get_model(self):
        self.model = getattr(__import__("models"), self.configs.model)(configs=self.configs)
        if self.configs.decoupling:
            head = copy.deepcopy(self.model.fc)
            self.model.fc = nn.Identity()
            self.model = getattr(__import__("models"), "BaseHeadSplit")(self.model, head)
        if not self.configs.parallel:
            self.model = self.model.to(self.configs.device)

    def load_train_data(self, shuffle=True):
        return DataLoader(
            self.load_data(self.train_file), self.configs.batch_size,
            drop_last=True, shuffle=shuffle)

    def load_test_data(self, shuffle=True):
        return DataLoader(
            self.load_data(self.test_file), self.configs.batch_size,
            drop_last=False, shuffle=shuffle)

    def test_metrics(self):
        testloader = self.load_test_data()
        self.test_samples = len(testloader.dataset)
        if self.configs.parallel: self.model.to(self.configs.device)
        self.model.eval()
        test_acc, losses = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.configs.device), y.to(self.configs.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                loss = self.loss(output, y)
                losses += loss.item() * y.shape[0]
        acc = test_acc / self.test_samples if self.test_samples > 0 else 0
        loss = losses / self.test_samples if self.test_samples > 0 else 0
        self.metrics["accs"].append(acc)
        self.metrics["losses_test"].append(loss)
        if self.configs.parallel: self.model.to("cpu")
        return acc, loss

    def train_metrics(self):
        trainloader = self.load_train_data()
        train_num = len(trainloader.dataset)
        if self.configs.parallel: self.model.to(self.configs.device)
        self.model.eval()
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.configs.device), y.to(self.configs.device)
                output = self.model(x)
                loss = self.loss(output, y)
                losses += loss.item() * y.shape[0]
        loss = losses / train_num if train_num > 0 else 0
        self.metrics["losses"].append(loss)
        if self.configs.parallel: self.model.to("cpu")
        return loss

    def _optim_step(self):
        self.optimizer.step()

    def train(self):

        self.trainloader = self.load_train_data()
        self.train_samples = len(self.trainloader.dataset)
        if self.configs.parallel: self.model.to(self.configs.device)
        self.model.train()
        start_time = time.time()
        print(f"Node {self.id} - Training with {self.configs.epochs} epochs and {self.configs.batch_size} batch size")
        for _ in range(self.configs.epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.configs.device), y.to(self.configs.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self._optim_step()
        train_time = time.time() - start_time
        if self.configs.parallel:
            self.model.to("cpu")
            return {
                "id": self.id, "model": self.model.state_dict(),
                "optimizer": self.optimizer, "train_time": train_time,
                "train_samples": self.train_samples,
            }
        self.metrics["train_time"].append(train_time)

    def aggregation(self):
        assert len(self.uploaded_models) > 0
        self.uploaded_weights = np.array(self.uploaded_weights) / np.sum(self.uploaded_weights)
        model = self.zero_parameters(self.uploaded_models[0])
        for w, node_model in zip(self.uploaded_weights, self.uploaded_models):
            for server_param, node_param in zip(model.parameters(), node_model.parameters()):
                server_param.data += node_param.data.clone() * w
        self.set_parameters(model=model)

    def set_parameters(self, model):
        for param1, param2 in zip(self.model.parameters(), model.parameters()):
            param1.data = param2.data.clone()

    def set_parameters1(self, state_dict):
        self.model.load_state_dict(state_dict)

