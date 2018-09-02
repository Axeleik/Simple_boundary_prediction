from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch


class blocksdataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, raw_list, gt_list, val=False, transform=None):

        assert (len(raw_list) == len(gt_list)), "raw and gt lists should be the same!"

        self.raw_list = raw_list
        self.gt_list = gt_list
        self.transform = transform
        self.val = val

    def __len__(self):
        return len(self.raw_list)

    def __getitem__(self, idx):

        raw = torch.tensor(self.raw_list[idx], dtype = torch.float32)
        gt = torch.tensor(self.gt_list[idx], dtype = torch.float32)

        if raw.max()>1 or raw.min()<0:
            raw /= 255

        if self.transform:
            raw = self.transform(raw)
            gt = self.transform(gt)

        return raw.unsqueeze(dim=0), gt.unsqueeze(dim=0)