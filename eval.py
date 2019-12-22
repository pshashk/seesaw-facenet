import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from io import BytesIO
from PIL import Image
import pickle


class FacePairs(torch.utils.data.Dataset):
    def __init__(self, path):
        self.x, self.y = pickle.load(open(path, "rb"), encoding="bytes")
        self.y = torch.tensor(self.y).int()
        self.transform = Compose(
            [ToTensor(), Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True)]
        )

    @staticmethod
    def loader(bin):
        return Image.open(BytesIO(bin)).convert("RGB")

    def __getitem__(self, index):
        return self.transform(self.loader(self.x[index]))

    def __len__(self):
        return len(self.x)


def accuracy(y_true, y_hat):
    y_true_sorted = y_true[y_hat.argsort()]
    fp = y_true_sorted.cumsum(0)
    fn = (1 - y_true_sorted).flip(0).cumsum(0).flip(0)
    acc = 1.0 - (fp + fn).min().float() / y_true.size(0)
    return acc


def eval(model, bin_path, batch_size=256):
    dataset = FacePairs(bin_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    emb = []
    device = next(model.parameters()).device
    for x in dataloader:
        with torch.no_grad():
            y = model(x.to(device))
            emb.append(y.cpu())

    emb = torch.cat(emb, 0)
    emb = torch.nn.functional.normalize(emb)
    y_hat = (emb[0::2, :] * emb[1::2, :]).sum(1).numpy()
    return accuracy(dataset.y, y_hat)


if __name__ == "__main__":
    import sys
    import os

    model_path, bin_dir = sys.argv[1], sys.argv[2]
    model = torch.jit.load(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    for bin_path in os.scandir(bin_dir):
        if bin_path.is_file() and bin_path.name.endswith(".bin"):
            acc = eval(model, bin_path.path)
            print(f"{bin_path.name[:-4]}: {acc:.2%}")
