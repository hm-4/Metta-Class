import torch
import torchvision

def torchvision_datasets_printing_format(data_set, data_set_name: str = "data"):
    
    string1 = f"\n\nData format: \n" \
            f"--------------------------------\n"\
            f"type({data_set_name}.data[0]) -> {type(data_set.data[0])}\n"\
            f"{data_set_name}.data[0].dtype -> {data_set.data[0].dtype}\n"\
            f"\n"\
            f"type({data_set_name}.targets) -> {type(data_set.targets)}\n"\
            f"{data_set_name}.targets.dtype -> {data_set.targets.dtype}\n"
    print(string1)
    
    string2 = f"\nAccess Data by indexing\n"\
            f"--------------------------------\n"\
        f"{data_set_name}[0] -> {data_set[0][0].shape, data_set[0][1].shape}\n"
    print(string2)
    string3 = f"\nmax and min values of {data_set_name}.data:\n"\
        f"--------------------------------\n"\
        f"\t[{torch.min(data_set.data), torch.max(data_set.data)}]\n"
    
    print(string3)
    
    print("\n================================\n")
    # train_data.data[0].dtype