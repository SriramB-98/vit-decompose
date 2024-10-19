from .cached_imgnet import CachedImageNet
import pickle, os

class PixelImageNet(CachedImageNet):
    def __init__(self, img_root, mask_root, img_transform=None, mask_transform=None):
        super().__init__(root=img_root, split='train', transform=img_transform)
        self.mask_root = mask_root
        self.mask_transform = mask_transform
        with open('./data_indices.pkl', 'rb') as fp:
            self.data_indices = pickle.load(fp)
#         self.samples = [(img_path, self.get_mask_path(img_path), l) for img_path, l in self.samples]
#         self.samples = [(ip, mp, l) for ip, mp, l in self.samples if os.path.exists(mp)]
        return
    
    def get_mask_path(self, img_path):
        mask_path = self.mask_root + '/'.join(img_path.split('/')[-2:])
        mask_path = mask_path[:-4] + 'png'
        return mask_path
    
    def __len__(self):
        return len(self.data_indices)
    
    def __getitem__(self, index: int):
        img_path, target = self.samples[self.data_indices[index]]
        mask_path = self.get_mask_path(img_path)
        if not os.path.exists(mask_path):
            return None
        sample = self.loader(img_path)
        mask = self.loader(mask_path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
#             mask = mask[:,:,:,0]
#             mask = mask.bool().float()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, mask, target