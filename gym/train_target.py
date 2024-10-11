# trainign loop for MLP models

from config import config
from data.scm_dataset import SCMDataset
from models.target_network import TargetNetwork
from torch import Tensor, no_grad
from torch.nn import MSELoss
from torch.optim import AdamW
from tqdm import tqdm


class TargetNetworkTrainer:
    """Class containing a model and an SCM dataset, providing train and eval methods."""

    def __init__(self):
        """Initialize model, dataset, loss and optimizer."""
        self.model = TargetNetwork(num_x=config.target.num_x, num_y=config.target.num_y)
        self.dataset = SCMDataset(num_x=config.target.num_x, num_y=config.target.num_y,
                                  batch_size=config.target.training.batch_size)
        self.loss_fn = MSELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=config.target.training.lr)

    def train(self):
        """Run training as specified in config."""
        self.model.train()
        for it, (x, y) in tqdm(enumerate(self.dataset)):
            self.optimizer.zero_grad()
            if it > config.target.training.iterations:
                break

            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)

            loss.backward()
            self.optimizer.step()

            if it % 100 == 0:
                # print(y)
                print(f"Loss after iteration {it}: {loss:.5f}")

    def evaluate(self) -> tuple[Tensor, Tensor]:
        """Compute prediction on one batch with model in eval mode."""
        self.model.eval()
        with no_grad():
            x, y = next(iter(self.dataset))
            y_pred = self.model(x)

        return y, y_pred

if __name__ == "__main__":
    TargetNetworkTrainer().train().evaluate()
