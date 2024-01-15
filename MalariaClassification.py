#!/usr/bin/env python
# coding: utf-8


import os

from torch_snippets import *



id2int = {'Parasitized': 0, 'Uninfected': 1}


from torchvision import transforms as T
trn_tfms = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ColorJitter(brightness=(0.9,1.05), # Randomly changes the 
                  #brightness, constrast and hue, in the range
            contrast = (0.95,1.05),
            saturation=(0.95,1.05),
                  hue=0.05),
    T.RandomAffine(5,translate=(0.01,0.1)), # Rotation and 
    # fraction of image width and height
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5],
               std=[0.5,0.5,0.5]),
                  
                  
    
])



val_tfms = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])


class MalariaImages(Dataset):
    
    def __init__(self,files,transform = None):
        self.files = files
        self.transform = transform
        logger.info(len(self))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,ix):
        fpath = self.files[ix]
        clss = os.path.basename(os.path.dirname(fpath)) 
        
        img = read(fpath,1)
        return img,clss
    
    def choose(self):
        return self[randint(len(self))]
    
    def collate_fn(self, batch):
        _imgs, classes = list(zip(*batch))

        # Transform images and add an extra dimension
        if self.transform:
            imgs = torch.stack([self.transform(img) for img in _imgs])
        else:
            imgs = torch.stack([img for img in _imgs])

        # Convert classes to tensor
        classes = torch.tensor([id2int[clss] for clss in classes])

        # Move to the specified device (like GPU, if available)
        imgs = imgs.to(device)
        classes = classes.to(device)

        return imgs, classes, _imgs


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
all_files = Glob('MalariaClassfication/archive/cell_images/*/*.png') # Makes list of all the fies
np.random.seed(10)
np.random.shuffle(all_files)

from sklearn.model_selection import train_test_split
trn_files,val_files = train_test_split(all_files,random_state=1)
trn_ds = MalariaImages(trn_files,transform=trn_tfms)
val_ds = MalariaImages(val_files,transform=val_tfms)

trn_dl = DataLoader(trn_ds,32,shuffle=True, collate_fn=trn_ds.collate_fn)
val_dl = DataLoader(val_ds,32,shuffle=True,collate_fn=val_ds.collate_fn)



def ConvBlock(ni,no):
    return nn.Sequential(
        nn.Dropout(0.2),
        nn.Conv2d(ni,no,kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(2)
    )

class MalariaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        ConvBlock(3,64),
        ConvBlock(64,64),
        ConvBlock(64,128),
        ConvBlock(128,256),
            ConvBlock(256,512),
            ConvBlock(512,64),
            nn.Flatten(),
            nn.Linear(256,256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256,len(id2int)) 
        )
        self.loss_fn=nn.CrossEntropyLoss()
    def forward(self,x):
            return self.model(x)
            
    def compute_metrices(self,preds,targets):
            loss = self.loss_fn(preds,targets)
            acc = (torch.max(preds,1)[1] == targets).float().mean()
            return loss,acc



def train_batch(model,data,optimizer,criterion):
    model.train()
    ims,labels,_ = data 
    #print(ims.shape)
    _preds = model(ims)
    optimizer.zero_grad()
    loss,acc = criterion(_preds,labels)
    loss.backward()
    optimizer.step()
    return loss.item(),acc.item()

@torch.no_grad()
def validate_batch(model,data,optimizer,criterion):
    model.eval()
    ims,labels,_ = data
    _preds = model(ims)
    loss,acc = criterion(_preds,labels)
    
    return loss.item(),acc.item()


# Model Training 

model = MalariaClassifier().to(device)
criterion = model.compute_metrices
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

n_epochs = 2
log = Report(n_epochs)
for ex in range(n_epochs):
    N = len(trn_dl)
    for bx,data in enumerate(trn_dl):
        loss,acc = train_batch(model,data,optimizer,criterion)
        log.record(ex+(bx+1)/N,trn_loss=loss,trn_acc=acc,end='\r')

    N = len(val_dl)
    for bx,data in enumerate(val_dl):
        loss,acc = validate_batch(model,data,optimizer,criterion)
        log.record(ex+(bx+1)/N,val_loss=loss,val_acc=acc)
        
    log.report_avgs(ex+1)



im2fmap=nn.Sequential(*(list(model.model[:5].children())+list(model.model[5][:2].children())))
model.model

# Cam Generation for Looking at how Conv layer is working
def im2gradCAM(x):
    model.eval()
    logits = model(x)
    heatmaps = []
    activations = im2fmap(x)
#    print(activations.shape)
    pred = logits.max(-1)[-1]
    model.zero_grad()
    logits[0,pred].backward(retain_graph=True)
    pooled_grads = model.model[-7][1].weight.grad.data.mean((0,2,3))
    for i in range(activations.shape[1]):
        activations[:,i,:,:] *= pooled_grads[i]
    heatmap = torch.mean(activations,dim=1)[0].cpu().detach()
    return heatmap, 'Uninfected' if pred.item() else 'Parasitized'
    


SZ = 128
def upsampleHeatmap(map, img):
    m,M = map.min(), map.max()
    map = 255 * ((map-m) / (M-m))
    map = np.uint8(map)
    map = cv2.resize(map, (SZ,SZ))
    map = cv2.applyColorMap(255-map, cv2.COLORMAP_JET)
    map = np.uint8(map)
    map = np.uint8(map*0.7 + img*0.3)
    return map


N = 20
_val_dl = DataLoader(val_ds, batch_size=N, shuffle=True, \
                     collate_fn=val_ds.collate_fn)
x,y,z = next(iter(_val_dl))

for i in range(N):
    image = resize(z[i], SZ)
    heatmap, pred = im2gradCAM(x[i:i+1])
    if(pred=='Uninfected'):
        continue
    heatmap = upsampleHeatmap(heatmap, image)
    subplots([image, heatmap], nc=2, figsize=(5,3), \
                suptitle=pred)



model.model[-7][1].weight.grad.data.mean((0,2,3)).shape

model.model





