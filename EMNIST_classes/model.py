import os
import time
from math import ceil

import numpy as np
import pandas as pd
import torch
from torch import nn, functional
from torch.utils.data import DataLoader, TensorDataset


class EMNIST_model:
    def __init__(self, number_of_classes, model, default_batch_size=64, default_model_path="EMNIST_model.pt", default_log_path="EMNIST_log.txt"):
        self.model = model
        self.number_of_classes = number_of_classes
        self.default_batch_size = default_batch_size
        self.default_model_path = default_model_path
        self.default_log_path = default_log_path

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model.to(self.device)

    def save_weights(self, path=None):
        if path is None:
            path = self.default_model_path
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path=None):
        if path is None:
            path = self.default_model_path
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))

    def predict(self, x, batch_size=0):
        if batch_size == 0:
            batch_size = self.default_batch_size
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        data_loader = DataLoader(x, batch_size=batch_size)
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                predictions.extend(self.model(batch).cpu().numpy().argmax(axis=1))
        return np.array(predictions)

    def predict_sinle_input(self, x):
        x = np.array([x])
        return self.predict(x)[0]

    def evaluate(self, x, y, verbose=True, loss_function=nn.CrossEntropyLoss, batch_size=0):
        if batch_size == 0:
            batch_size = self.default_batch_size
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        data_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)

        loss = loss_function()
        correct = 0
        total = 0
        mean_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                batch_x, batch_y = batch
                predictions_onehot = self.model(batch_x)
                loss_value = loss(predictions_onehot, batch_y)
                _, predicted = torch.max(predictions_onehot.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                mean_loss += loss_value.item()
                num_batches += 1
        accuracy = float(correct) / total
        mean_loss /= num_batches
        if verbose:
            print(f"Accuracy: {accuracy}, mean loss: {mean_loss}, number of examples: {total}")

        return accuracy, mean_loss


    def train(self, x, y, validation_x, validation_y, max_epochs_number=100, stop_after_no_improvement=5, batch_size=0,
              loss_function=nn.CrossEntropyLoss, optimizer=torch.optim.Adam, learning_rate=0.001, verbose=True,
              validation_metric_accuracy=True, model_path=None, log_path=None):
        log_df = pd.DataFrame(columns=["epoch", "train_accuracy", "train_loss", "validation_accuracy", "validation_loss", "time", "number_of_training_examples", "number_of_validation_examples", "saved_during_training"])

        if batch_size == 0:
            batch_size = self.default_batch_size

        if model_path is None:
            model_path = self.default_model_path

        if log_path is None:
            log_path = self.default_log_path

        train_size = len(x)
        validation_size = len(y)
        train_num_batches = int(ceil(train_size / batch_size))
        validation_num_batches = int(ceil(validation_size / batch_size))

        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        validation_x = torch.tensor(validation_x, dtype=torch.float32).to(self.device)
        validation_y = torch.tensor(validation_y, dtype=torch.long).to(self.device)

        dataset_train = TensorDataset(x, y)
        dataset_validation = TensorDataset(validation_x, validation_y)
        data_loader_train_set = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        data_loader_validation_set = DataLoader(dataset_validation, batch_size=batch_size)

        optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        loss = loss_function()

        if verbose:
            print(f"Training on {train_size} examples, validating on {validation_size} examples,\n\tbatch size: {batch_size}, max epochs number: {max_epochs_number},\n\tstop after no improvement: {stop_after_no_improvement}, learning rate: {learning_rate} \n\tvalidation metric: {'accuracy' if validation_metric_accuracy else 'loss'}, model path: {model_path} \n\tdevice: {self.device}, loss function: {loss.__class__.__name__},\n\toptimizer: {optimizer.__class__.__name__}, log path: {log_path}\n\n")

        last_improvement_epoch = 0
        last_best_result = 0 if validation_metric_accuracy else 100000000.0
        for epoch in range(max_epochs_number):
            self.model.train()
            seconds_begin = time.time()
            correct_train = 0
            correct_validation = 0
            mean_training_loss = 0
            mean_validation_loss = 0

            for batch in data_loader_train_set:
                batch_x, batch_y = batch
                predictions_onehot = self.model(batch_x)
                train_loss_value = loss(predictions_onehot, batch_y)
                train_loss_value.backward()
                optimizer.step()
                optimizer.zero_grad()
                # accuracy calculation
                _, predicted = torch.max(predictions_onehot.data, 1)
                correct_train += (predicted == batch_y).sum().item()
                mean_training_loss += train_loss_value.item()

            self.model.eval()
            with torch.no_grad():
                for batch in data_loader_validation_set:
                    batch_x, batch_y = batch
                    predictions_onehot = self.model(batch_x)
                    validation_loss_value = loss(predictions_onehot, batch_y)
                    # accuracy calculation
                    _, predicted = torch.max(predictions_onehot.data, 1)
                    correct_validation += (predicted == batch_y).sum().item()
                    mean_validation_loss += validation_loss_value.item()

            seconds_end = time.time()

            mean_training_loss /= train_num_batches
            mean_validation_loss /= validation_num_batches

            train_accuracy = float(correct_train) / train_size
            validation_accuracy = float(correct_validation) / validation_size


            if validation_metric_accuracy:
                current_result = validation_accuracy
                is_better = current_result > last_best_result
            else:
                current_result = mean_validation_loss
                is_better = current_result < last_best_result

            new_row_df = pd.DataFrame([{
                "epoch": epoch,
                "train_accuracy": train_accuracy,
                "train_loss": mean_training_loss,
                "validation_accuracy": validation_accuracy,
                "validation_loss": mean_validation_loss,
                "time": (seconds_end - seconds_begin),
                "number_of_training_examples": len(x),
                "number_of_validation_examples": len(validation_x),
                "saved_during_training": 1 if is_better else 0
            }])
            log_df = pd.concat([log_df, new_row_df], ignore_index=True)
            log_df.to_csv(log_path, index=False)

            if verbose:
                print(f"\nEpoch: {epoch}, time: {(seconds_end - seconds_begin):.3f} seconds\n\taccuracy: {train_accuracy:.3f}, loss: {mean_training_loss:.3f}\n\tvalidation_accuracy: {validation_accuracy:.3f}, validation loss: {mean_validation_loss:.3f}")
                if is_better:
                    print(
                        f"\n\timproved from {last_best_result:.3f} (epoch {last_improvement_epoch}) to {current_result:.3f} (epoch {epoch})")
                else:
                    print(f"\n\tnot improved from {last_best_result:.3f} (epoch {last_improvement_epoch})")

            if is_better:
                last_best_result = current_result
                last_improvement_epoch = epoch
                self.save_weights(model_path)
            elif epoch - last_improvement_epoch > stop_after_no_improvement:
                break

        self.load_weights(model_path)

        if verbose:
            print(f"\n\nTraining finished. Best result: {last_best_result:.3f} (epoch {last_improvement_epoch})")

    @staticmethod
    def load_logs(log_path):
        return pd.read_csv(log_path)

    @staticmethod
    def load_logs_accuracy_losses_saves(log_path):
        data_frame = pd.read_csv(log_path)
        return data_frame["train_accuracy"].values, data_frame["validation_accuracy"].values, data_frame["train_loss"].values, data_frame["validation_loss"].values, data_frame["saved_during_training"].values



