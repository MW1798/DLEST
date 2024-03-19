# Mengqi Wu  hereby modifies this code under the terms of the Apache License, Version 2.0 
# (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of 
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is 
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and limitations under the License.
#
# This file was originally authored by Stanislav Pidhorskyi and is subject to the Apache License, Version 2.0.
# The modifications to this file were made by Mengqi Wu.
# Nature of modification:
# - Added interface for saving final results to npy file
# ==============================================================================

# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import copy
import shutil
import torch.utils.data
import torchvision as tv
from glob import glob
from torchvision.utils import save_image
import random
import sys
import numpy as np
import os
print(os.getcwd())
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch import optim, autograd
from torchvision import transforms, utils

from submit import _create_run_dir_local, _copy_dir
sys.path.append('./ALAE')
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
from ebm import LatentEBM
from logger import Logger
import distributed as dist
import tqdm
from PIL import Image
from sgld import SGLD
import re
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm, trange
from skimage.metrics import structural_similarity as ssim
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter('runs/ebm_training')

lreq.use_implicit_lreq.set(True)

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
	'.tif', '.TIF', '.tiff', '.TIFF','.npy','.nii'
]


def ema(model1, model2, decay=0.999):
	par1 = dict(model1.named_parameters())
	par2 = dict(model2.named_parameters())

	for k in par1.keys():
		par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				path = os.path.join(root, fname)
				images.append(path)
	return images[:min(max_dataset_size, len(images))]


def default_loader(path):
	return Image.open(path).convert('RGB')

def npy_loader(path):
	return None


class ImageFolder(data.Dataset):

	def __init__(self, root, transform=None, return_paths=False,
				 loader=default_loader):
		imgs = make_dataset(root)
		if len(imgs) == 0:
			raise (RuntimeError("Found 0 images in: " + root + "\n"
															   "Supported image extensions are: " + ",".join(
				IMG_EXTENSIONS)))

		self.root = root
		self.imgs = imgs
		self.transform = transform
		self.return_paths = return_paths
		self.loader = loader

	def __getitem__(self, index):
		path = self.imgs[index]
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)
		if self.return_paths:
			return img, path
		else:
			return img

	def __len__(self):
		return len(self.imgs)

class MRIDataset_processed(data.Dataset):
	FILETYPE = ['nii','npy']
	def __init__(self,image_dir:str,filetype:str,slice_range:int,annotation_file=None): 
		assert Path(image_dir).is_dir(), f'{image_dir} is not a valid directory'
		if annotation_file is not None: assert Path(annotation_file).is_file(), f'{annotation_file} is not a valid file'
		assert filetype in MRIDataset_processed.FILETYPE, f'filetype must be in {MRIDataset_processed.FILETYPE}'
		
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
			
		
		if not 'SRPBS' in str(self.img_dir):
			label = self.img_labels.iloc[idx,1]
			subject_id = self.img_labels.iloc[idx,0]
			label = torch.tensor(label,device='cpu')
		else:
			label = self.img_labels.iloc[idx,2]
			subject_id = self.img_labels.iloc[idx,1]

  
		img_volume = torch.tensor(img_volume,device='cpu')
		# print('min, max, mean after normalization: ',[img_volume.min(),img_volume.max(),img_volume.mean()])
		# print(img_volume.shape)
		# print(type(img_volume))
		return img_volume, subject_id,label
	

def requires_grad(model, flag=True):
	for p in model.parameters():
		p.requires_grad = flag

def langvin_sampler(model, x, langevin_steps=20, lr=1.0, sigma=0e-2, return_seq=False):
	x_orig = x.clone().detach()
	x = x.clone().detach()
	# content_criteria = torch.nn.MSELoss() # l2 loss
	content_criteria = torch.nn.L1Loss() # l1 loss

	x.requires_grad_(True)
	sgd = SGLD([x], lr=lr, std_dev=sigma)
	sequence = torch.zeros_like(x).unsqueeze(0).repeat(langevin_steps, 1, 1)
	for k in range(langevin_steps):
		sequence[k] = x.data
		model.zero_grad()
		sgd.zero_grad()
		energy = model(x).sum()
  
		loss_content = content_criteria(x_orig,x) 
		loss = 0.1*(-energy)+100*loss_content
		loss.backward()
		sgd.step()

	if return_seq:
		return sequence
	else:
		return x.clone().detach(),loss

def load_ae(cfg, logger):
	torch.cuda.set_device(0)
	model = Model(
		startf=cfg.MODEL.START_CHANNEL_COUNT,
		layer_count=cfg.MODEL.LAYER_COUNT,
		maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
		latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
		truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
		truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
		mapping_layers=cfg.MODEL.MAPPING_LAYERS,
		channels=cfg.MODEL.CHANNELS,
		generator=cfg.MODEL.GENERATOR,
		encoder=cfg.MODEL.ENCODER)
	model.cuda()
	model.eval()
	model.requires_grad_(False)


	decoder = model.decoder
	encoder = model.encoder
	mapping_tl = model.mapping_tl
	mapping_fl = model.mapping_fl
	dlatent_avg = model.dlatent_avg

	logger.info("Trainable parameters generator:")
	count_parameters(decoder)

	logger.info("Trainable parameters discriminator:")
	count_parameters(encoder)

	arguments = dict()
	arguments["iteration"] = 0

	model_dict = {
		'discriminator_s': encoder,
		'generator_s': decoder,
		'mapping_tl_s': mapping_tl,
		'mapping_fl_s': mapping_fl,
		'dlatent_avg': dlatent_avg
	}

	checkpointer = Checkpointer(cfg,
								model_dict,
								{},
								logger=logger,
								save=False)

	extra_checkpoint_data = checkpointer.load()

	model.eval()

	layer_count = cfg.MODEL.LAYER_COUNT

	path = cfg.DATASET.SAMPLES_PATH
	im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)
	return model

def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	else:
		shutil.rmtree(log_dir)
		os.makedirs(log_dir)
	return log_dir

def encode(model, x, cfg):

	Z, _ = model.encode(x, cfg.MODEL.LAYER_COUNT - 1, 1)
	return Z.squeeze(1)


def decode(model, x, cfg):
	layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
	ones = torch.ones(layer_idx.shape, dtype=torch.float32)
	coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
	return model.decoder(x, cfg.MODEL.LAYER_COUNT - 1, 1, noise=True)



def test(cfg, logger):
	# Create save path
	data_root = cfg.DATA.ROOT
	prefix = cfg.DATA.NAME + "-" + cfg.DATA.SOURCE + '2' + cfg.DATA.TARGET
	save_path = os.path.join(data_root, prefix)

	img_dir = Path(save_path) /f'{cfg.DATASET.ROOT}_img_2_{cfg.DATA.TARGET}'
	if not img_dir.exists():
		os.makedirs(img_dir)

	sys.stdout = Logger(os.path.join(save_path, 'log.txt'))
 
	weight_base = Path('results') / prefix
	ebm_weight_pt = weight_base/ cfg.EBM.WEIGHTS / 'model_best.pt'
	ckp = torch.load(ebm_weight_pt)
	print(f'load EBM weight from: {ebm_weight_pt}')
	print(cfg)
	

	ae = load_ae(cfg, logger).to('cuda')

	device = 'cuda'
	
	data_root = Path(cfg.DATA.ROOT) 
	dataset_root = data_root / cfg.DATASET.ROOT 
	dataset_path = dataset_root / cfg.DATA.NAME 
	if cfg.DATASET.ROOT == 'val':
		dataset_path = dataset_root / 'val_quasiraw_10slices'
	print(dataset_path)
 
	label_root = dataset_root / f'{cfg.DATASET.ROOT}_labels'
	source_label = label_root / f'{cfg.DATA.SOURCE}.tsv'
	target_label = label_root / f'{cfg.DATA.TARGET}.tsv'
	

	source_dataset = MRIDataset_processed(dataset_path,'npy',5,annotation_file=source_label)
	source_sampler = dist.data_sampler(source_dataset, shuffle=True, distributed=False)
	source_loader = DataLoader(source_dataset, batch_size=cfg.DATA.BATCH, sampler=source_sampler, num_workers=0, generator=torch.Generator(device='cuda'))
	target_dataset = MRIDataset_processed(dataset_path,'npy',5, annotation_file=target_label)
	target_sampler = dist.data_sampler(target_dataset, shuffle=True, distributed=False)
	target_loader = DataLoader(
		target_dataset, batch_size=cfg.DATA.BATCH, sampler=target_sampler, num_workers=0, generator=torch.Generator(device='cuda')
	)
	assert len(source_loader) > len(target_loader),'source loader size < target loader size, change cycle() in batch for loop!'
	tqdm.write(f'source size: {len(source_loader.dataset)}, target size: {len(target_loader.dataset)}')


	latent_ebm = LatentEBM(latent_dim=512, n_layer=cfg.EBM.LAYER, n_hidden=cfg.EBM.HIDDEN).to(device)
	
	latent_ebm.load_state_dict(ckp['model_state_dict'])
	latent_ebm.eval()

	iterations = 0
	nrow = min(cfg.DATA.BATCH, 12)
	batch_size = cfg.DATA.BATCH


	src_sbj_count = 0
	tar_sbj_count = 0
	
	for batch_source,sbj_id_src,site_lb in tqdm(source_loader,desc='source batches'):
		recon_slice = torch.tensor([])
		for slice in range(batch_source.shape[1]):
			iterations += 1
			source_slices = batch_source[:,slice,:,:].unsqueeze(1).float() # (b,1,256,256)
			source_slices = source_slices.to(device)

			source_latent = encode(ae, source_slices, cfg)
			source_latent = source_latent.squeeze() # (b,512)
			
			source_latent_q,_ = langvin_sampler(latent_ebm, source_latent.clone().detach(),
											langevin_steps=cfg.LANGEVIN.STEP, lr=cfg.LANGEVIN.LR,)
			with torch.no_grad():
				recon_latent = source_latent_q.unsqueeze(1).repeat(1, ae.mapping_fl.num_layers, 1)
				recon = decode(ae, recon_latent, cfg)
				recon_slice = torch.cat((recon_slice,recon),dim=1)
		
		for img_vol,id,site in zip(recon_slice,sbj_id_src,site_lb): 
			if not type(id)==str:
				id = str(id.item())
			if not type(site) == str:
				site = str(site.item())
			fn = str(img_dir / f'{id}_{site}.npy')
			np.save(fn,img_vol.to('cpu').numpy())
			src_sbj_count += 1
	
	for batch_target,sbj_id_target,site_lb in tqdm(target_loader,desc='target batches'):
		for img_vol, id, site in zip(batch_target,sbj_id_target,site_lb):
			if not type(id)==str:
				id = str(id.item())
			if not type(site) == str:
				site = str(site.item())
			fn = str(img_dir / f'{id}_{site}.npy') 
			np.save(fn,img_vol.to('cpu').numpy())
			tar_sbj_count += 1

	assert len(source_loader.dataset) == src_sbj_count and len(target_loader.dataset) == tar_sbj_count, 'subject count miss match!'
	tqdm.write(f'saved {src_sbj_count} source subjects, {tar_sbj_count} target subjects, subject counts matched!')
		

if __name__ == "__main__":
	gpu_count = 1
	run(test, get_cfg_defaults(), description='Image-Translation', default_config='ALAE/configs/SRPBS_tst.yaml',
		world_size=gpu_count, write_log=False)
	# run(train, get_cfg_defaults(), description='Image-Translation', default_config='ALAE\configs\celeba-hq1024.yaml',
	# 	world_size=gpu_count, write_log=False)
