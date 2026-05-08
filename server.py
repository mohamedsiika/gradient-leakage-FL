import torch
from torch.utils.data import DataLoader
from dataset import SemanticSegmentationDataset

def get_eval_fn(model, val_image, val_mask):
    def evaluate(server_round: int, parameters, config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        val_loader = DataLoader(SemanticSegmentationDataset(val_image, val_mask), batch_size=1)
        criterion = torch.nn.CrossEntropyLoss()
        model.eval()
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                out = model(img)
                loss = criterion(out, mask)
        return float(loss.item()), {}
    return evaluate