import flwr as fl
from client import FlowerClient
from server import get_eval_fn
import torch
import segmentation_models_pytorch as smp

if __name__ == "__main__":
    train_image = "fold1_1171_Adrenal-gland_patch0.png"
    train_mask = "fold1_1171_Adrenal-gland_patch0.npy"
    val_image = "fold1_1171_Adrenal-gland_patch0.png"
    val_mask = "fold1_1171_Adrenal-gland_patch0.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=6,
            activation=None,
    ).to(device)


    def client_fn(cid: str):
        # For a true FL simulation, change data selection logic per cid
        return FlowerClient(model, train_image, train_mask)

    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_eval_fn(model, val_image, val_mask)
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,  # Increase for more simulated clients
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )