import torch
from torch.utils.data import DataLoader
import flwr as fl
from dataset import SemanticSegmentationDataset

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_image, train_mask):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.train_loader = DataLoader(
            SemanticSegmentationDataset(train_image, train_mask), batch_size=1
        )

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(100):  # one epoch
            for img, mask in self.train_loader:
                img, mask = img.to(self.device), mask.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(img)
                loss = self.criterion(out, mask)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            for img, mask in self.train_loader:
                img, mask = img.to(self.device), mask.to(self.device)
                out = self.model(img)
                loss = self.criterion(out, mask)
        return float(loss.item()), len(self.train_loader.dataset), {}