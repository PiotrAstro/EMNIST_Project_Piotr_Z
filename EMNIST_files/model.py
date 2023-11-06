import numpy as np
import torch
from torch import nn, functional
from torch.utils.data import DataLoader, TensorDataset


class EMNIST_model:
    def __init__(self, x_size, y_size, number_of_classes, model, default_batch_size=64, default_model_path="EMNIST_model.pt"):
        self.model = model
        self.x_size = x_size
        self.y_size = y_size
        self.number_of_classes = number_of_classes
        self.default_batch_size = default_batch_size
        self.default_model_path = default_model_path

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model.to(self.device)

    def save(self, path=None):
        if path is None:
            path = self.default_model_path
        torch.save(self.model.state_dict(), path)

    def load(self, path=None):
        if path is None:
            path = self.default_model_path
        self.model.load_state_dict(torch.load(path))

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        data_loader = DataLoader(x, batch_size=self.default_batch_size)
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                predictions.extend(self.model(batch).cpu().numpy().max(axis=1))
        return predictions

    def predict_sinle_input(self, x):
        x = np.array([x])
        return self.predict(x)[0]

    def evaluate(self, x, y, verbose=True, loss_function=nn.CrossEntropyLoss):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        data_loader = DataLoader(TensorDataset(x, y), batch_size=self.default_batch_size)

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


    def train(self, x, y, validation_x, validation_y, max_epochs_number=100, stop_after_no_improvement=5, batch_size=64,
              loss_function=nn.CrossEntropyLoss, optimizer=torch.optim.Adam, learning_rate=0.001, verbose=True,
              validation_metric_accuracy=True):
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
        last_improvement_epoch = 0
        last_best_result = 0 if validation_metric_accuracy else 100000000.0
        for epoch in range(max_epochs_number):
            self.model.train()
            correct_train = 0
            total_train = 0
            correct_validation = 0
            total_validation = 0
            mean_training_loss = 0
            mean_validation_loss = 0
            num_batches_train = 0
            num_batches_validation = 0
            for batch in data_loader_train_set:
                batch_x, batch_y = batch
                predictions_onehot = self.model(batch_x)
                train_loss_value = loss(predictions_onehot, batch_y)
                train_loss_value.backward()
                optimizer.step()
                optimizer.zero_grad()
                # accuracy calculation
                _, predicted = torch.max(predictions_onehot.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()
                mean_training_loss += train_loss_value.item()
                num_batches_train += 1

            self.model.eval()
            with torch.no_grad():
                for batch in data_loader_validation_set:
                    batch_x, batch_y = batch
                    predictions_onehot = self.model(batch_x)
                    validation_loss_value = loss(predictions_onehot, batch_y)
                    # accuracy calculation
                    _, predicted = torch.max(predictions_onehot.data, 1)
                    total_validation += batch_y.size(0)
                    correct_validation += (predicted == batch_y).sum().item()
                    mean_validation_loss += validation_loss_value.item()
                    num_batches_validation += 1

            mean_training_loss /= num_batches_train
            mean_validation_loss /= num_batches_validation

            train_accuracy = float(correct_train) / total_train
            validation_accuracy = float(correct_validation) / total_validation

            if validation_metric_accuracy:
                current_result = validation_accuracy
                is_better = current_result > last_best_result
            else:
                current_result = mean_validation_loss
                is_better = current_result < last_best_result

            if verbose:
                print(f"\nEpoch: {epoch}\n\taccuracy: {train_accuracy}, loss: {mean_training_loss}, num: {x.size()}\n\tvalidation_accuracy: {validation_accuracy}, validation loss: {mean_validation_loss}, validation_num: {validation_x.size()}")
                if is_better:
                    print(
                        f"\n\timproved from {last_best_result} (epoch {last_improvement_epoch}) to {current_result} (epoch {epoch})")
                else:
                    print(f"\n\tnot improved from {last_best_result} (epoch {last_improvement_epoch})")

            if is_better:
                last_best_result = current_result
                last_improvement_epoch = epoch
                self.save()
            elif epoch - last_improvement_epoch > stop_after_no_improvement:
                break

        if verbose:
            print(f"\n\nTraining finished. Best result: {last_best_result} (epoch {last_improvement_epoch})")





