
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
# from torchdata.datapipes.iter import IterableWrapper, Mapper
import os
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):

        self.output_size = output_size

    def __call__(self, sample):
        image= sample
        output_size=self.output_size
        h, w = image.shape[:2]
        
        end_h=h-output_size
        end_w=w-output_size
    
        h_s=np.random.randint(0,end_h)
        w_s=np.random.randint(0,end_w)
    
        h_e=h_s+output_size
        w_e=w_s+output_size

        image=image[h_s:h_e,w_s:w_e,:]
     

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
 
        
        return torch.tensor(image,dtype=torch.float32)

class ToBlur(object):
    def __init__(self,kernel_size):
        self.kernel_size=kernel_size
    def __call__(self,sample):
        landmarks = sample
        image=torchvision.transforms.functional.gaussian_blur(sample,kernel_size=self.kernel_size,sigma=(0.1,1.5))
        return {'image': image,
                'landmarks': landmarks}

class SubUp(object):
    def __init__(self,factor):
        self.factor=factor
    def __call__(self,sample):
        image, landmarks = sample['image'], sample['landmarks']
        size=landmarks.shape[1:]
        factor=self.factor
        image=image[:,::factor,::factor]
        
        #comment out three lines below to use original low_quality image without upsampling
        upsample=torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        image=upsample.forward(image)
        image=torch.clamp(image,min=0,max=1)
                
        return {'image': image,
                'landmarks':landmarks}                  
    


class Dataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dirs,repeat, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        paths=list()
        for root_dir in root_dirs:
            list_path=[os.path.join(root_dir,fname) for fname in os.listdir(root_dir)]
            paths=paths+list_path
        
        self.paths = (paths*repeat)
        self.root_dirs = root_dirs
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.paths[idx]
        sample = plt.imread(img_name) # of shape HxWx3
        if sample.shape[-1]>3:
            sample=sample[:,:,0:3]

        # if gray scale=>duplicate 3 times
        if sample.shape[-1]==1:
            sample=np.stack((sample[0],sample[0],sample[0]),axis=-1)
                
        if self.transform:
            sample = self.transform(sample)

        return sample

# PyTorch models inherit from torch.nn.Module
class SuperReso(nn.Module):
    def __init__(self):
        super(SuperReso, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 11,padding=1)
        self.batchnorm1=nn.BatchNorm2d(128)
        #self.pool = nn.MaxPool2d(2, 2).to()
        self.conv2 = nn.Conv2d(128,64 ,3,padding=1)
        self.batchnorm2=nn.BatchNorm2d(64)
        
        self.conv3=nn.Conv2d(64,3,5)

    def forward(self, x):
        x=self.conv1(x)
        x=F.leaky_relu(self.batchnorm1(x),negative_slope=0.2)
        x=self.conv2(x)
        x=F.leaky_relu(self.batchnorm2(x),negative_slope=0.2)
        x=self.conv3(x)
        return x
    

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.001)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.001)
        torch.nn.init.constant_(m.bias, 0)


def customized_loss(outputs,labels):
    out_size=outputs.shape[-1]
    
    crop_func=torchvision.transforms.CenterCrop(out_size)
    labels=crop_func.forward(labels)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(outputs, labels)
    return loss


def train_one_epoch(num_batch,model,training_loader,optimizer,customized_loss,DEVICE):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs= data['image'].to(DEVICE)
        
        labels=data['landmarks'].to(DEVICE)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        
        # Compute the loss and its gradients
        loss = customized_loss(outputs, labels)
        print("loss: ",loss)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
     
        if i % num_batch == (num_batch-1):
            last_loss = running_loss / num_batch # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss



def convert_image_batch(model,one_batch,DEVICE):
    """Given one batch from validation_loader, 
     display 3 images per image: hr, sr, interpolation """
    with torch.no_grad():
        hr=one_batch['landmarks'].to(DEVICE)
        image=one_batch['image'].to(DEVICE)
        model.to(DEVICE)

        sr=model(image)
        output_size=sr.shape[-1]
        hr=torchvision.transforms.CenterCrop(output_size).forward(hr)
        image=torchvision.transforms.CenterCrop(output_size).forward(image)
        # return hr,sr,image

        #permute to shape BxHxWx3 

        hr=torch.permute(hr,(0,2,3,1))
        sr=torch.clamp(torch.permute(sr,(0,2,3,1)),min=0,max=1)
        image=torch.permute(image,(0,2,3,1))

        hr=np.array(hr.detach().cpu())
        sr=np.array(sr.detach().cpu())
        image=np.array(image.detach().cpu())
        

    return (hr,sr,image)

def display_per(one_batch):
    hr,sr,image=one_batch[0],one_batch[1],one_batch[2]
    batch_size=len(hr)
    plt.figure(figsize=(10,10)) # specifying the overall grid size
    start=np.random.randint(0,batch_size-3)

    for i in range(3*3):
        plt.subplot(3,3,i+1)    # the number of images in the grid is 5*5 (25)
        trace=i%3
        idx=int((i-trace)/3)+start
        if trace==0:
            img=hr[idx]
        elif trace==1:
            img=sr[idx]
        else:
            img=image[idx]  
        
        plt.imshow(img)

    plt.show()


def super_reso(model,img_path):
    # Given one low-res img, apply model to turn it into sr one
    name_file=img_path.split('/')[-1]
    name,ex=name_file.split('.')[0],name_file.split('.')[-1]

    img=plt.imread(img_path)[:,:,0:3]
    size=(img.shape[0]*2,img.shape[1]*2)

    img=torch.tensor(img,dtype=torch.float32)
    img=torch.permute(img,(2,0,1))
    img=torch.unsqueeze(img,dim=0)
    


    upsample=torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    bi=upsample.forward(img)
    bi=torch.clamp(bi,min=0,max=1)

  
    sr=model(bi).detach().cpu()
    sr=torch.clamp(sr,min=0,max=1)
    sr=np.array(torch.permute(sr[0],(1,2,0)))
    sr_file=Image.fromarray((sr*255.0).astype(np.uint8))
    sr_file.save('save/{}_sr.{}'.format(name,ex))

    bi=np.array(torch.permute(bi[0],(1,2,0)))
    bi_file=Image.fromarray((bi*255.0).astype(np.uint8))
    bi_file.save('save/{}_bi.{}'.format(name,ex))
    









    