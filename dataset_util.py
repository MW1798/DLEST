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
	


