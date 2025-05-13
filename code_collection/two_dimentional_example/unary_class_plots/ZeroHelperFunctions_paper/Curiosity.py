import torch

class CuriosityRover:
    def __init__(self, device):
        self.learned_anomaly_models = []
        self.names_of_anomalies = []
        self.device = device
        self.temp_probabilites = None
        
    def add_model_for_an_anomaly(self, new_learned_anomaly_model, name_given_to_anomaly):
        self.learned_anomaly_models.append(new_learned_anomaly_model)
        self.names_of_anomalies.append(name_given_to_anomaly)

    def predictions(self, x):
        self.temp_probabilites = None
        x = x.to(self.device)
        self.temp_probabilites = torch.zeros((len(x), len(self.learned_anomaly_models)))

        for m, model in enumerate(self.learned_anomaly_models):
            model.to(self.device).float()
            model.eval()
            # print(x.shape, type(x[0]), type(x))
            with torch.inference_mode():
                logits = model(x)
                # print(logits)
                self.temp_probabilites[:, m] = torch.sigmoid(logits).squeeze(dim=1)

        predictions = torch.argmax(self.temp_probabilites, dim=1)
        # The comment `# min because the zero class label is 1` in the code snippet is explaining the
        # reason for using `torch.argmin` function in the `predictions` method of the `CuriosityRover`
        # class.
        # min because the zero class label is 1
        return predictions