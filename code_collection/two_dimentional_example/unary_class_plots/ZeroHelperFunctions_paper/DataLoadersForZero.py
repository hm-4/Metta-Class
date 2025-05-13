import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from ZeroHelperFunctions_paper.zero_class_data_generator import ZeroClassDataset
from ZeroHelperFunctions_paper.CustomDataset import CustomDataset


class DataLoadersForZero:
    def __init__(self, train_data, test_data, data_shape_of_data_point=None):
        self.train_data = train_data
        self.test_data = test_data
        self.data_shape = data_shape_of_data_point # to generalize.
        
        self._zero_data_for_train = None
        self._zero_data_for_test = None
        self.zero_data_for_printing = None
        self.zero_labels_for_printing = None
        
        self._train_data_save_to_generate_zero=None
        # for boundary plotting.
        self.combined_train_data = None
        self.combined_train_targets = None
        self.combined_test_data = None
        self.combined_test_targets = None
        
        self._train0_data = None
        self._test0_data = None
        
        self.train_dataloader = None
        self.train0_dataloader = None
        self.test_dataloader = None
        self.test0_dataloader = None
        self.zero_dataloader = None
    
    def generate_zero_class_dataloader(self,
                                    n_zeros,
                                    batch_size,
                                    label):
        """ send the train_data before it is wrapped. 
        """
        zero_data = ZeroClassDataset(num_samples=n_zeros,
                                        data_shape=self.data_shape,
                                        label=label,
                                        data=self._train_data_save_to_generate_zero.data)
        self.zero_dataloader = DataLoader(zero_data,
                                        batch_size=batch_size,
                                        shuffle=False)
    
    
    def _generate_zero_class_Datasets(self,
                                    n_train_zeros: int, 
                                    n_test_zeros: int,
                                    label_for_zero):
        self._zero_data_for_train = ZeroClassDataset(num_samples=n_train_zeros,
                                                    data_shape=self.data_shape,
                                                    label=label_for_zero,
                                                    data=self.train_data.data)
        
        self._zero_data_for_test = ZeroClassDataset(num_samples=n_test_zeros,
                                                data_shape=self.data_shape,
                                                label=label_for_zero,
                                                data=self.test_data.data)
        self.zero_data_for_printing = self._zero_data_for_test.data
        self.zero_labels_for_printing = self._zero_data_for_test.targets
        self.combined_train_data = torch.cat((self.train_data.data, 
                                            self._zero_data_for_train.data),
                                            dim=0)
        self.combined_train_targets = torch.cat((self.train_data.targets,
                                                self._zero_data_for_train.targets))
        self.combined_test_data = torch.cat((self.test_data.data, 
                                            self._zero_data_for_test.data),
                                            dim=0)
        self.combined_test_targets = torch.cat((self.test_data.targets,
                                                self._zero_data_for_test.targets))
        
    def make_dataloaders(self, batch_size, 
                        n_train_zeros, 
                        n_test_zeros,
                        label_for_zero):
        # generate all zero classdata
        self._generate_zero_class_Datasets(n_train_zeros=n_train_zeros,
                                        n_test_zeros=n_test_zeros,
                                        label_for_zero=label_for_zero)
        # wrap data
        self._wrap_train_and_test_datasets()
        # concatenate
        self._combine_zero()
        
        self.train_dataloader = DataLoader(self.train_data,
                                        batch_size=batch_size,
                                        shuffle=True)
        self.train0_dataloader = DataLoader(self._train0_data,
                                            batch_size=batch_size,
                                            shuffle=True)
        self.test_dataloader = DataLoader(self.test_data,
                                        batch_size=batch_size,
                                        shuffle=False)
        self.test0_dataloader = DataLoader(self._test0_data,
                                        batch_size=batch_size,
                                        shuffle=False)
        
    def _combine_zero(self):
        self._train0_data = ConcatDataset([self.train_data, self._zero_data_for_train])
        self._test0_data = ConcatDataset([self.test_data, self._zero_data_for_test])
    
    def _wrap_train_and_test_datasets(self):
        self._train_data_save_to_generate_zero = self.train_data
        self.train_data = CustomDataset(self.train_data.data, self.train_data.targets)
        self.test_data = CustomDataset(self.test_data.data, self.test_data.targets)
        