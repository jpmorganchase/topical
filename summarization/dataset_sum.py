import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, all_source_ids,all_source_mask,all_target_ids=None,all_target_mask=None):
        self.all_source_ids = all_source_ids
        self.all_source_mask = all_source_mask
        self.all_target_ids = all_target_ids
        self.all_target_mask = all_target_mask

    def __len__(self):
        return len(self.all_target_ids)

    def __getitem__(self, idx):
        if self.all_target_ids is not None and self.all_target_mask is not None:
            return self.all_source_ids[idx], self.all_source_mask[idx], self.all_target_ids[idx], self.all_target_mask[idx]
        else:
            return self.all_source_ids[idx], self.all_source_mask[idx]