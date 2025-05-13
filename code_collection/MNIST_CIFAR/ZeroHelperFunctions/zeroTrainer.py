import torch
from torch import nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader


import sys
import os
import gc
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


from ZeroHelperFunctions.zero_class_images_generator_mnist import ZeroClassDataset
from ZeroHelperFunctions.Accuracy_fn import accuracy_fn

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
                device):
        self.model = model
        self.num_classes = number_of_non_zero_classes
        self.label_of_zero_class = label_of_zero_class
        
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.pf_dl = purity_fact_dl
        self.of_dl = zero_dl
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.device = device
        
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
    
    def train(self, epochs):
        # set up the storing
        self._initialize_metrics(epochs)
        print(f"Training...(epochs: {epochs})")
        for epoch in range(epochs):
            print(f"Epoch: {epoch}\n----------------")
            self._run_epoch(epoch)
    
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
            # print(X.shape)
            self.optimizer.zero_grad()
            
            y_logits = self.model(X)
            if self.num_classes == 0:
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
            # print(y_pred.shape, y.shape)
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
                if self.num_classes == 0:
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
        if self.num_classes > 0:
            return logits.argmax(dim=1)
        return torch.round(torch.sigmoid(logits.squeeze(dim=1)))
    