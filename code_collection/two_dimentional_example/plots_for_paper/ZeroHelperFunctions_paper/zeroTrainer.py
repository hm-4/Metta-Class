import torch
from torch import nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import matplotlib.pyplot as plt

import sys
import os
import gc
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


from ZeroHelperFunctions_paper.zero_class_data_generator import ZeroClassDataset
from ZeroHelperFunctions_paper.Accuracy_fn import accuracy_fn
from ZeroHelperFunctions_paper.helper_functions import plot_decision_boundary, plot_decision_boundary_non_zero
from ZeroHelperFunctions_paper import plots
import numpy as np

class ZeroTrainer:
    def __init__(self,
                model,
                number_of_non_zero_classes,
                train_dl,
                test_dl,
                purity_fact_dl,
                zero_dl,
                loss_fn,
                optimizer,
                label_of_zero_class,
                device,
                plot_data,
                plot_targets, 
                zero_exists=1):
        self.model = model
        self.num_classes = number_of_non_zero_classes
        self.num_classes_including_zero = number_of_non_zero_classes + zero_exists 
        self.label_of_zero_class = label_of_zero_class
        
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.pf_dl = purity_fact_dl
        self.of_dl = zero_dl
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.device = device
        self.plot_data = plot_data
        self.plot_targets = plot_targets
        
        self.train_loss = None
        self.train_acc = None
        
        self.test_loss = None
        self.test_acc = None
        
        self.purities = None
        self.occupancy = None
    
    def _initialize_metrics(self, epochs):
        self.train_loss = torch.zeros(epochs)
        self.train_acc = torch.zeros(epochs)
        
        self.test_loss = torch.zeros(epochs)
        self.test_acc = torch.zeros(epochs)
        
        self.purities = torch.zeros(epochs, self.num_classes).to(self.device)
        self.occupancy = torch.zeros(epochs)
    
    def train(self, epochs, image_dir="training_images"):
        # set up the storing
        # Create directory to save images
        # image_dir = "training_images"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        self._initialize_metrics(epochs)
        print(f"Training...(epochs: {epochs})")
        for epoch in range(epochs):
            print(f"Epoch: {epoch}\n----------------")
            self._run_epoch(epoch)
            # plot model and corresponding graphs. 
            fig = plt.figure(figsize=(16, 12))
            # fig.suptitle(f'Epoch-{epoch}', fontsize=16)
            
            plt.subplot(2, 2, 1)
            plt.title("Decision Boundaries")
            print("num_classes", self.num_classes_including_zero, self.num_classes_including_zero==4)
            if self.num_classes_including_zero == 4:
                plot_decision_boundary(self.model, self.plot_data, self.plot_targets, n_classes=self.num_classes_including_zero)
            else:
                plot_decision_boundary_non_zero(self.model, self.plot_data, self.plot_targets, n_classes=self.num_classes_including_zero)
            
            x = np.arange(epochs)
            plt.subplot(2, 2, 2)
            plt.title("accuracy")
            train_acc = self.train_acc.cpu().numpy()
            test_acc = self.test_acc.cpu().numpy()
            masked_train_acc = np.where(train_acc == 0, np.nan, train_acc)
            masked_test_acc = np.where(test_acc==0, np.nan, test_acc)
            plt.plot(x, masked_train_acc, marker='o', markersize=2, linewidth=0.5, label='Train accuracy w/ zero class')
            plt.plot(x, masked_test_acc, marker='x', markersize=2, linewidth=0.5, label='Test accuracy w/o zero class')
            plt.legend()
            plt.ylim(-5, 105)
            plt.grid(True)
            
            
            plt.subplot(2, 2, 3)
            plt.title("Purity Factors")
            tensor = self.purities.cpu().numpy()
            n, c = tensor.shape
            masked_tensor = np.where(tensor == 0, np.nan, tensor)
            for i in range(c):
                plt.plot(x, masked_tensor[:, i], marker='o', markersize=2, linewidth=0.5, label=f'class-{i}')
            plt.legend()
            plt.ylim(-0.1, 1.1)
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.title("Occupancy Factor")
            occ = self.occupancy.cpu().numpy()
            masked_occ = np.where(occ==0, np.nan, occ)
            plt.plot(x, masked_occ, marker='o', markersize=2, linewidth=0.5)
            
            # plt.legend()
            plt.ylim(-0.1, 1.1)
            plt.grid(True)
            plt.tight_layout()
            # Save the figure as an image file in the specified directory
            plt.savefig(os.path.join(image_dir, f'epoch_{epoch}.pgf'))
            plt.close(fig)
            
    def _run_epoch(self, epoch):
        self._epoch_train(epoch)
        
        self._epoch_test(epoch)
        self._epoch_pf(epoch)
        self._epoch_of(epoch)
    
    def _epoch_train(self, epoch):
        self.model.to(self.device)
        train_loss, train_acc = 0, 0
        for X, y in self.train_dl:
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            y_logits = self.model(X)
            if self.num_classes == 1:
                loss = self.loss_fn(y_logits.squeeze(dim=1), y.float())
            else:
                # print(y_logits.shape, y.shape)
                loss = self.loss_fn(y_logits, y)
            # print(y_logits.shape, y.shape)
            # loss = self.loss_fn(y_logits, y)
            train_loss += loss
            
            loss.backward()
            self.optimizer.step()
            
            y_pred = self._logit_predictions(y_logits)
            train_acc += accuracy_fn(true=y, pred=y_pred)
            
        num_batches = len(self.train_dl)
        self.train_loss[epoch] = train_loss/num_batches
        self.train_acc[epoch] = train_acc/num_batches
    
    def _epoch_test(self, epoch):
        self.model.to(self.device)
        self.model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for X, y in self.test_dl:
                X, y = X.to(self.device), y.to(self.device)
                
                y_logits = self.model(X)
                if self.num_classes == 1:
                    loss = self.loss_fn(y_logits.squeeze(dim=1), y.float())
                else:
                    loss = self.loss_fn(y_logits, y)
                test_loss += loss
                
                y_pred = self._logit_predictions(y_logits)
                test_acc += accuracy_fn(true=y, pred=y_pred)
                
        num_batches = len(self.test_dl)
        self.test_loss[epoch] = test_loss/num_batches
        self.test_acc[epoch] = test_acc/num_batches
        
    def _epoch_pf(self, epoch):
        self.model.to(self.device)
        self.model.eval()
        predictions = torch.zeros(self.num_classes).to(self.device)
        predictions_and_labels = torch.zeros(self.num_classes).to(self.device)
        with torch.inference_mode():
            for X, y in self.pf_dl:
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
                y_pred = self._logit_predictions(y_logits)
                preds, preds_labels = self._batch_pf(y_pred, y)
                predictions += preds
                predictions_and_labels += preds_labels
                
        mask = predictions != 0
        
        self.purities[epoch][mask] = predictions_and_labels[mask]/predictions[mask]
    
    def _batch_pf(self, y_pred, y):
        predictions = torch.zeros(self.num_classes).to(self.device)
        predictions_and_labels = torch.zeros(self.num_classes).to(self.device)
        for cl in range(self.num_classes):
            pred_class_or_not = torch.where(y_pred==cl, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))
            label_class_or_not = torch.where(y==cl, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))
            num_pred_labels = torch.dot(pred_class_or_not.float(), label_class_or_not.float())
            num_pred = torch.sum(pred_class_or_not)
            predictions[cl] += num_pred
            predictions_and_labels[cl] += num_pred_labels
        return predictions, predictions_and_labels
    
    def _epoch_of(self, epoch):
        self.model.to(self.device)
        self.model.eval()
        occ_points = 0
        total = 0
        with torch.inference_mode():
            for X, y in self.of_dl:
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
                y_pred = self._logit_predictions(y_logits)
                occ_points += torch.sum(y_pred != self.label_of_zero_class).item()
                total += y_pred.numel()
        self.occupancy[epoch] = occ_points/total
    
    # def _accuracy_fn(self, true, pred):
    #     correct = torch.eq(true, pred).sum().item()
    #     acc = (correct / len(pred)) * 100
    #     return acc
    
    def _logit_predictions(self, logits):
        if self.num_classes > 1:
            return logits.argmax(dim=1)
        return torch.round(torch.sigmoid(logits.squeeze(dim=1)))
    