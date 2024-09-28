import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from matplotlib import cm
import matplotlib
import model_io
import utils
from models import UnetAdaptiveBins

def yield_file(in_file):
    f = open(in_file)
    buf = f.read()
    f.close()
    for b in buf.split('\n'):
        if b.startswith('v '):
            yield ['v', [float(x) for x in b.split(" ")[1:]]]
        elif b.startswith('f '):
            triangle = b.split(' ')[1:]
            # -1 as .obj is base 1 but the Data class expects base 0 indices
            yield ['f', [[int(i) - 1 for i in t.split("/")] for t in triangle]]
        else:
            yield ['', ""]


def read_obj(in_file):
    vertices = []
    faces = []

    for k, v in yield_file(in_file):
        if k == 'v':
            vertices.append(v)
        elif k == 'f':
            for i in v:
                faces.append(i)

    if not len(faces) or not len(vertices):
        return None

    pos = np.array(vertices, dtype=np.float) #torch.tensor(vertices, dtype=torch.float)
    print("initial pos",pos.shape)
    pos = np.expand_dims(pos, axis=-1)#torch.unsqueeze(pos, dim=-1)
    # pos = torch.unsqueeze(pos, dim=-1) 
    pos = np.resize(pos,(64,40,3))#torch..resize_(3,40,64)
    pos = np.repeat(pos,10,axis=0)#pos.repeat(1,1,12,10)
    pos = np.repeat(pos,12,axis=1)
    pos = np.repeat(pos,1,axis=2)
    # face = torch.tensor(faces, dtype=torch.float).t().contiguous()
    # print("initial face",face.shape)
    # face = torch.unsqueeze(face, dim=-1) 
    # face = torch.unsqueeze(face, dim=-1) 
    # face = face.resize_(1,128,240,10)
    # d = torch.concat((face,pos),3).cuda()
    d = pos #cuda()
    print(pos.shape)
    # print(face.shape)
    # print(d.shape)
    return d

# data = read_obj("/content/AdaBins/pip4.obj")

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480)):
        # image = image.resize(target_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            print("pic",pic.shape)
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class InferenceHelper:
    def __init__(self, dataset='nyu', device='cuda:0'):
        self.toTensor = ToTensor()
        self.device = device
        if dataset == 'nyu':
            self.min_depth = 1e-3
            self.max_depth = 10
            self.saving_factor = 1000  # used to save in 16 bit
            model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = "/content/drive/MyDrive/AdaBins_nyu.pt"
        elif dataset == 'kitti':
            self.min_depth = 1e-3
            self.max_depth = 80
            self.saving_factor = 1000
            model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = "./pretrained/AdaBins_kitti.pt"
        else:
            raise ValueError("dataset can be either 'nyu' or 'kitti' but got {}".format(dataset))

        model, _, _ = model_io.load_checkpoint(pretrained_path, model)
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict_pil(self, pil_image ,test_d ,out_d , visualized=False):
        pil_image = np.resize(pil_image,(640, 480,3)) #pil_image.resize((640, 480))
        img = np.asarray(pil_image) / 255.
        
        img = self.toTensor(img).unsqueeze(0).float().to(self.device)
        bin_centers, pred = self.predict(img, test_d, out_d)

        if visualized:
            viz = utils.colorize(torch.from_numpy(pred).unsqueeze(0), vmin=None, vmax=None, cmap='magma')
            # pred = np.asarray(pred*1000, dtype='uint16')
            viz = Image.fromarray(viz)
            return bin_centers, pred, viz
        return bin_centers, pred

    @torch.no_grad()
    def predict(self, image ,test_dir , out_dir):
        bins, pred = self.model(image)
        pred = np.clip(pred.cpu().numpy(), self.min_depth, self.max_depth)

        # Flip
        image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(self.device)
        pred_lr = self.model(image)[-1]
        pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], self.min_depth, self.max_depth)

        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)
        final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:],
                                          mode='bilinear', align_corners=True).cpu().numpy()

        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[np.isinf(final)] = self.max_depth
        final[np.isnan(final)] = self.min_depth

        centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        centers = centers.cpu().squeeze().numpy()
        centers = centers[centers > self.min_depth]
        centers = centers[centers < self.max_depth]
        
        final_t = (final * self.saving_factor).astype('uint16')
        basename = os.path.basename(test_dir).split('.')[0]
        save_path = os.path.join(out_dir, basename + ".png")

        viz = utils.colorize(torch.from_numpy(final), vmin=None, vmax=None, cmap='magma')
        Image.fromarray(viz).save(save_path)
       
        # Image.fromarray(final_t.squeeze()).save(save_path)

        return centers, final

    @torch.no_grad()
    def predict_dir(self, test_dir, out_dir):
        print("HOOOOOOOOOOO")
        os.makedirs(out_dir, exist_ok=True)
        transform = ToTensor()
        all_files = test_dir #glob.glob(os.path.join(test_dir, "*"))
        self.model.eval()
        for f in tqdm(all_files):
            image = np.asarray(Image.open(f), dtype='float32') / 255.
            print("HIIIIIIIII")
            image = transform(image).unsqueeze(0).to(self.device)

            centers, final = self.predict(image)
            # final = final.squeeze().cpu().numpy()

            final = (final * self.saving_factor).astype('uint16')
            basename = os.path.basename(f).split('.')[0]
            save_path = os.path.join(out_dir, basename + ".png")

            Image.fromarray(final).save(save_path)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from time import time
    test_direction = "/content/AdaBins/test_imgs/pipe4.png"
    out_direction = "/content/AdaBins"
    img = Image.open(test_direction)
    print("img",np.array(img).shape)
    start = time()
    inferHelper = InferenceHelper()
    centers, pred = inferHelper.predict_pil(img,test_direction,out_direction)
    print(f"took :{time() - start}s")
    from google.colab.patches import cv2_imshow
    cv2_imshow(pred.squeeze())

    plt.imshow(pred.squeeze(), cmap='magma_r')
    plt.show()

