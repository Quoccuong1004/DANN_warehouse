import torch.utils.data as data
from PIL import Image
import os
import torch
import pandas as pd

class GetLoader(data.Dataset):

  def __init__(self,csv_file,root_dir,transform=None) :
    self.annotations = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self,index):
    img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
    image = Image.open(img_path)
    if image.mode != "RGB":
      image = image.convert("RGB")
    y_label = torch.tensor(int(self.annotations.iloc[index,1]))

    if self.transform :
      image = self.transform(image)

    return( image,y_label)
