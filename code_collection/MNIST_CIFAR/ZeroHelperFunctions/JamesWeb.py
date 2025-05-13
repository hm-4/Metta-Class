import torch
import torch.nn.functional as F

class JamesWeb:
    def __init__(self, device):
        self.learned_models = []
        self.model_names = []
        self.model_zero_class_labels = []
        self.device = device
        self.temp_probabilities = None
        
    def add_model(self, model, model_name, model_zero_class_label):
        self.learned_models.append(model)
        self.model_names.append(model_name)
        self.model_zero_class_labels.append(model_zero_class_label)
    
    def convert_labels(self, labels, model_name):
        labels = labels.unsqueeze(dim=1)
        try:
            index = self.model_names.index(model_name)
            print(f"{model_name} is in the list at index {index}.")
        except ValueError:
            print(f"{model_name} is not in the list.")
        new_row = torch.full((labels.size(0), 1), index, dtype=torch.int64)
        resultant_tensor = torch.cat((new_row, labels), dim=1)
        return resultant_tensor
    
    def predictions(self, x):
        x = x.to(self.device)
        self.temp_probabilities = torch.zeros((len(x), len(self.learned_models), 3), dtype=torch.float32)

        for m, model in enumerate(self.learned_models):
            model.to(self.device)
            model.eval()
            with torch.inference_mode():
                logits = model(x)
                # print("\n\nlogits\n", logits)
                cls = logits.argmax(dim=1)
                indices_of_zero = torch.nonzero(cls == self.model_zero_class_labels[m], 
                                                as_tuple=True)
                # print("\n\ncls\n", cls)
                probabilities = F.softmax(logits, dim=1)
                # print("\n\nprobabilities\n", probabilities)
                max_probability, _ = torch.max(probabilities, dim=1)
                
                # print("\n\n max_probability\n", max_probability)
                max_probability[indices_of_zero] = 0  # handling the autozero class assignment.
                # print("\n\n max_probability2\n", max_probability)
                self.temp_probabilities[:, m, 0] = m
                self.temp_probabilities[:, m, 1] = cls
                self.temp_probabilities[:, m, 2] = max_probability
                # print("\n\nself.temp_probabilities\n", self.temp_probabilities)
        # Find the highest probability (m, cls) pair
        max_prob_indices = self.temp_probabilities[:, :, 2].argmax(dim=1)
        # print("\n\nmax_prob_indices\n", max_prob_indices)
        predictions = self.temp_probabilities[torch.arange(len(x)), max_prob_indices, :2]
        # print("\n\npredictioins\n", predictions)
        return predictions.to(torch.int64)
                
                
