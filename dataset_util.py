from io import BytesIO
import torchvision as tv, torchvision.transforms as tr
import lmdb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch


class MultiResolutionDataset(Dataset):
	def __init__(self, path, transform, resolution=8):
		self.env = lmdb.open(
			path,
			max_readers=32,
			readonly=True,
			lock=False,
			readahead=False,
			meminit=False,
		)

		if not self.env:
			raise IOError('Cannot open lmdb dataset', path)

		with self.env.begin(write=False) as txn:
			self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

		self.resolution = resolution
		self.transform = transform

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		with self.env.begin(write=False) as txn:
			key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
			img_bytes = txn.get(key)

		buffer = BytesIO(img_bytes)
		img = Image.open(buffer)
		img = self.transform(img)
		# print(type(img))
		return img

class MRIDataset(Dataset):
	FILETYPE = ['nii','npy']
	def __init__(self,image_dir:str,filetype:str,slice_range:int,transform=None,annotation_file=None): 
		assert Path(image_dir).is_dir(), f'{image_dir} is not a valid directory'
		if annotation_file is not None: assert Path(annotation_file).is_file(), f'{annotation_file} is not a valid file'
		assert filetype in MRIDataset.FILETYPE, f'filetype must be in {MRIDataset.FILETYPE}'
		
		self.transfrom = transform
		self.slice_range = slice_range
		self.img_dir = image_dir
		self.img_labels = pd.read_csv(annotation_file,sep='\t')
		self.filetype = filetype
		self.length = len(self.img_labels)
  
	def __len__(self):
		return self.length

	def __getitem__(self,idx):
		img_path_base = Path(self.img_dir)
		img_path_full = str(next(img_path_base.glob(f'*{self.img_labels.iloc[idx,0]}*.npy')))
		if self.filetype == 'npy':
			img_volume = np.load(img_path_full).squeeze()
			img_volume = np.flip(np.swapaxes(img_volume,0,1),0) # swap x,y, flip image to orient brain upright
			plr,ptb = (256-img_volume.shape[1])/2, (256-img_volume.shape[0])/2
			L,T,R,B = int(np.floor(plr)),int(np.floor(ptb)),int(np.ceil(plr)),int(np.ceil(ptb))
			transform = tr.Compose(
						[
							tr.ToTensor(),
							tr.Pad((L,T,R,B)) # L,T,R,B
						]
					)
   
		label = self.img_labels.iloc[idx,1]
  
		# getting rid of artifact around edges	
		img_volume[img_volume<0] = 0.0
  

		# normalize to [0,1]
		img_volume = (img_volume-img_volume.min())/(img_volume.max()-img_volume.min())

		x,y,z = img_volume.shape
		ranges = np.arange(z//2-self.slice_range,z//2+self.slice_range)
		img_slices = img_volume[:,:,ranges]

		label = torch.tensor(label,device='cpu')
		if self.transfrom:
			img_slices = transform(img_slices)

		return img_slices,label

class MRIDataset_processed(Dataset):
	FILETYPE = ['nii','npy']
	def __init__(self,image_dir:str,filetype:str,slice_range:int,transform=None,annotation_file=None): 
		assert Path(image_dir).is_dir(), f'{image_dir} is not a valid directory'
		if annotation_file is not None: assert Path(annotation_file).is_file(), f'{annotation_file} is not a valid file'
		assert filetype in MRIDataset_processed.FILETYPE, f'filetype must be in {MRIDataset_processed.FILETYPE}'
		
		self.transfrom = transform
		self.slice_range = slice_range
		self.img_dir = image_dir
		self.img_labels = pd.read_csv(annotation_file,sep='\t')
		self.filetype = filetype
		self.length = len(self.img_labels)

	def __len__(self):
		return self.length

	def __getitem__(self,idx):
		img_path_base = Path(self.img_dir)
		img_path_full = str(next(img_path_base.glob(f'*{self.img_labels.iloc[idx,0]}*.npy')))

		img_volume = np.load(img_path_full)
			
		label = self.img_labels.iloc[idx,1]	
		label = torch.tensor(label,device='cpu')
		img_volume = torch.tensor(img_volume,device='cpu')

		return img_volume,label
	


