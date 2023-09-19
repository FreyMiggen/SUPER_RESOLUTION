import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
import os
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import skimage
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
        image = sample # of shape HxWx3
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

class ToYcbcr(object):
    """Convert rgb image to Ycbcr image and keep the Y channel to feed to the SRNW. CbCr channels are stored separately.
    Output will be tensor of shape 1xHxW"""
    def __call__(self,sample):
        # if is grayscale=>no transform needed
        if sample.shape[-1]==1:
            return sample
        else:

            image=skimage.color.rgb2ycbcr(sample,channel_axis=-1) # of shape HxWx3

            image=torch.tensor(image,dtype=torch.float32)

            #reshape image to 3xHxW
            image=image.permute(2,0,1) 
            #only keep the y_channel
            image=torch.unsqueeze(image[0],dim=0)
            # scale input image to range [0,1]
            image=(image-16.0)/235.0

            return image


class ToBlur(object):
    def __init__(self,kernel_size):
        self.kernel_size=kernel_size
    def __call__(self,sample):
        
        
        image=torchvision.transforms.functional.gaussian_blur(sample,kernel_size=self.kernel_size,sigma=(0.1,1.5))
        
        image=torch.clamp(image,min=0,max=1)
        return {'image': image,
                'landmarks': sample}

class SubUp(object):
    def __init__(self,factor):
        self.factor=factor
    def __call__(self,sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        factor=self.factor
        image=image[:,::factor,::factor]
        
        #comment out three lines below to use original low_quality image without upsampling
        # upsample=torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        # image=upsample.forward(image)
        # image=torch.clamp(image,min=-1,max=1)
                
        return {'image': image,
                'landmarks':landmarks}                                                              

"""   some transformation classes for viewing the performance of model on real images"""


class ToYcbcrDisplay(object):
    """Convert rgb image to Ycbcr image 
    Output will be tensor of shape 3xHxW for colored img, 1xHxW for grayscale ones"""
    def __call__(self,sample):
        if sample.shape[-1]==1:
                        
            sample=torch.tensor(sample,dtype=torch.float32)
            
            #reshape image to 1xHxW
            image=sample.permute(2,0,1) 

            return sample
        else:
            image=skimage.color.rgb2ycbcr(sample,channel_axis=-1) # of shape HxWx3
            
            image=torch.tensor(image,dtype=torch.float32)
            
            #reshape image to 3xHxW
            image=image.permute(2,0,1) 

            return image


class ToBlurDisplay(object):
    def __init__(self,kernel_size):
        self.kernel_size=kernel_size
    def __call__(self,sample):
        label=sample
        
        image=torchvision.transforms.functional.gaussian_blur(sample,kernel_size=self.kernel_size,sigma=(0.1,1.5))
        if image.shape[0]==1:
            y_channel=np.expand_dims((image[0]-16.0)/235.0,axis=0)
            cb_channel=image[0]
            cr_channel=image[0]
        else:
            y_channel=np.expand_dims((image[0]-16.0)/235.0,axis=0) # of shape 1xHxW
            cb_channel=np.expand_dims(image[1],axis=0)
            cr_channel=np.expand_dims(image[2],axis=0)
        
        
        return {'y_channel': y_channel,
                'cb_channel':cb_channel,
                'cr_channel':cr_channel,
                'label': label}

class ToSubDisplay(object):
    def __init__(self,factor):
        self.factor=factor
    def __call__(self,sample):
        y_channel,cb_channel,cr_channel=sample['y_channel'],sample['cb_channel'],sample['cr_channel']
        label=sample['label']
        
        factor=self.factor
        y_channel=y_channel[:,::factor,::factor]
        cb_channel=cb_channel[:,::factor,::factor]
        cr_channel=cr_channel[:,::factor,::factor]
        
        return {'y_channel': y_channel,
                'cb_channel':cb_channel,
                'cr_channel':cr_channel,
                'label': label}
    


""" Finish transformation for display performance"""

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
        
        self.paths = paths*repeat
        self.root_dirs = root_dirs
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.paths[idx]
        sample = plt.imread(img_name)
        if sample.shape[-1]>3:
            sample=sample[:,:,0:3]
        # of shape HxWx3
                
        if self.transform:
            sample = self.transform(sample)

        return sample



class SuperReso(nn.Module):
    def __init__(self):
        super(SuperReso,self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 5,padding='same')
        # nn.init.normal_(self.conv1.weight,mean=0,std=0.001)
        self.batchnorm1=nn.BatchNorm2d(64)
        self.prelu1=nn.PReLU(init=0.2)
        
        self.conv2 = nn.Conv2d(64,32 ,3,padding='same')
        # nn.init.normal_(self.conv1.weight,mean=0,std=0.001)
        self.batchnorm2=nn.BatchNorm2d(32)
        self.prelu2=nn.PReLU(init=0.2)
        
        self.conv3=nn.Conv2d(32,4,3,padding='same')
        # nn.init.normal_(self.conv1.weight,mean=0,std=0.001)
        
        self.subpixel=nn.PixelShuffle(2)

    def forward(self, x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.prelu1(x)
        x=self.conv2(x)
        x=self.batchnorm2(x)
        x=self.prelu2(x)
        x=self.conv3(x)
        x=self.subpixel(x)
        
        return x
        
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.001)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.001)
        torch.nn.init.constant_(m.bias, 0)


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



#given a batch , produce result from SRNW and compare with label images
def assemble_img(batch,model,factor=2):
    y_channel=batch['y_channel'] # Bx1xHxW
    cb_channel=batch['cb_channel']
    cr_channel=batch['cr_channel']
    label=batch['label']
    
    
    # pass y_channel
    yc=model(y_channel).detach().cpu() # of shape Bx1xHxW
    yc=yc*235.0+16.0
    output_size=(y_channel.shape[2]*factor,y_channel.shape[3]*factor)
    trans=torchvision.transforms.Resize(output_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    
    cb=trans.forward(cb_channel)
    cr=trans.forward(cr_channel)
    cy=trans.forward(y_channel*235.0+16.0)
    
    cb=torch.clamp(cb,min=16.0,max=240.0)
    cr=torch.clamp(cr,min=16.0,max=240.0)
    cy=torch.clamp(cy,min=16.0,max=235.0)
    yc=torch.clamp(yc,min=16.0,max=235.0)
    
    
    sr_img=torch.concatenate((yc,cb,cr),dim=1)
    
    bi_img=torch.concatenate((cy,cb,cr),dim=1)
    

    return bi_img,sr_img,batch['label']


#given a batch of input image (torch tensor) in ycbcr space of shape Bx3xHxW=> rgb of shape HxWx3
def convert2rgb(img):
    img=img.permute(0,2,3,1)
    img=np.array(img,dtype=np.float32)
    result=map(skimage.color.ycbcr2rgb,list(img))
    
    return list(result)

def conv(img):
    img=img.permute(0,2,3,1)
    img=np.array(img,dtype=np.float32)
    return img
    
