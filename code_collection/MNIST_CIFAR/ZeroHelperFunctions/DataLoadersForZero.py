import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from ZeroHelperFunctions.zero_class_images_generator_mnist import ZeroClassDataset
from ZeroHelperFunctions.show_image import show_one_grayscale_image, show_one_color_image
from ZeroHelperFunctions.CustomDataset import CustomDataset


class DataLoadersForZero:
    def __init__(self, train_data, test_data, image_shape=None):
        self.train_data = train_data
        self.test_data = test_data
        self.image_shape = image_shape
        
        self._zero_data_for_train = None
        self._zero_data_for_test = None
        
        self._train_data_save_to_generate_zero=None
        
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
                                    label=10):
        """ send the train_data before it is wrapped. 
        """
        zero_data = ZeroClassDataset(num_samples=n_zeros,
                                        image_shape=self.image_shape,
                                        label=label)
        self.zero_dataloader = DataLoader(zero_data,
                                        batch_size=batch_size,
                                        shuffle=False)
    
    def _generate_zero_class_Datasets(self, 
                                    n_train_zeros: int,
                                    n_test_zeros: int,
                                    label_for_zero):
        # 1. check the compatibility of data .
        # 1.1 generate zero class data
        self._zero_data_for_train = ZeroClassDataset(num_samples=n_train_zeros,
                                                    image_shape=self.image_shape,
                                                    label=label_for_zero,)
        
        self._zero_data_for_test = ZeroClassDataset(num_samples=n_test_zeros,
                                                image_shape=self.image_shape,
                                                label=label_for_zero)
        # 2. wrap it
        # it is wrapped using the __getitem__ method
        
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
        
    # methods for diagnosing. 
    def describe_zero_class_data(self):
        """check order, it has to be called after make_dataloaders
        """
        print("\nzero_class_data_for_train.data:")
        print("shapes: ", self._zero_data_for_train.data.shape)
        print("types: ", type(self._zero_data_for_train.data))
        print("dtype: ", self._zero_data_for_train.data.dtype)
        print("\nzero_class_data_for_train.targets:")
        print("shapes: ", self._zero_data_for_train.targets.shape)
        print("types: ", type(self._zero_data_for_train.targets))
        print("dtype: ", self._zero_data_for_train.targets.dtype)
        print("\n")
        print("zero_class_data_for_test.data:")
        print("shapes: ", self._zero_data_for_test.data.shape)
        print("types: ", type(self._zero_data_for_test.data))
        print("dtype: ", self._zero_data_for_test.data.dtype)
        print("\nzero_class_data_for_test.targets:")
        print("shapes: ", self._zero_data_for_test.targets.shape)
        print("types: ", type(self._zero_data_for_test.targets))
        print("dtype: ", self._zero_data_for_test.targets.dtype)
    
    def check_elements_of_zero_class_if_tensors(self):
        self._check_elements_of_dataset_if_tensors(self._zero_data_for_train)
    
    def describe_train_data(self):
        """check order, it has to called before make_dataloaders
        """
        print("\ntrain_data.data and test_data.data:")
        print("shapes: ", self.train_data.data.shape, self.test_data.data.shape)
        print("types: ", type(self.train_data.data), type(self.test_data.data))
        print("dtype: ", self.train_data.data.dtype, self.test_data.data.dtype)
        
        print("\ntrain_data.targets and test_data.targets:")
        print("shapes: ", self.train_data.targets.shape, self.test_data.targets.shape)
        print("types: ", type(self.train_data.targets), type(self.test_data.targets))
        print("dtype: ", self.train_data.targets.dtype, self.test_data.targets.dtype)
    
    def check_elements_of_train_data_if_tensors(self):
        self._check_elements_of_dataset_if_tensors(self.train_data)
    
    def _check_elements_of_dataset_if_tensors(self, dataset):
        for i in range(len(dataset)):
            image, label = dataset[i]
            if not isinstance(image, torch.Tensor) or not isinstance(label, torch.Tensor):
                print(f"Non-tensor element found at index {i}: {type(image)}, {type(label)}")
            if i % 1000 == 0:
                print(f"Checked {i} elements")
    
    def check_dataloader(self, data_loader):
        for images, labels in data_loader:
            print(images.shape, images.dtype, labels.shape, labels.dtype)
            
    def show_border_images_of_combined_data(self, boarder):
        image1, label1 = self._train0_data[boarder-1000]
        image2, label2 = self._train0_data[boarder+1000]
        print("shape of image of data: ", image1.shape)
        print("shape of image of zero: ", image2.shape)
        # print(image1, image2)
        if self.image_shape[0] == 1:
            show_one_grayscale_image(image1)
            show_one_grayscale_image(image2)
        else:
            show_one_color_image(image1)
            show_one_color_image(image2)

