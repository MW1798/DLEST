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
# - Retained the main functionality of EBM.
# - Added reverse SGLD for cycle consistency loss
# - Added latent content loss
# - Modified the dataloader for paired training
# - Modified the interface for visualizing training samples and epoch statistics.
# ==============================================================================
# Original copyright text follows:
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
from torchvision import transforms, utils, models

from dummy import read_single_image
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
from kornia.losses import ssim_loss,psnr_loss

from content_style_loss import calculate_losses

from itertools import cycle
from torch.utils.tensorboard import SummaryWriter


# RUN_NAME = 'Syn_KPM_tst1'
RUN_NAME = 'Syn_Site17_tst4'

writer=SummaryWriter(f'runs/{RUN_NAME}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.vgg19(weights='DEFAULT').features.to(device).eval()

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
			
		label = self.img_labels.iloc[idx,1]	
		# label = torch.tensor(label,device='cpu')
		img_volume = torch.tensor(img_volume,device='cpu')
		# print('min, max, mean after normalization: ',[img_volume.min(),img_volume.max(),img_volume.mean()])
		# print(img_volume.shape)
		# print(type(img_volume))
		return img_volume
	

def requires_grad(model, flag=True):
	for p in model.parameters():
		p.requires_grad = flag

def langvin_sampler(model, x,y, langevin_steps=20, lr=1.0, sigma=0e-2, return_seq=False,target_slice = None,ae=None,cfg=None,pix=False):    
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

		if pix:
			# content_criteria = ssim
			recon_latent = x.unsqueeze(1).repeat(1, ae.mapping_fl.num_layers, 1)
			recon_slices = decode(ae, recon_latent, cfg) 
   
			org_latent = x_orig.unsqueeze(1).repeat(1, ae.mapping_fl.num_layers, 1)
			org_slices = decode(ae,org_latent,cfg)

			closs, sloss = calculate_losses(cnn,target_slice.repeat(1,3,1,1),org_slices.repeat(1,3,1,1),recon_slices.repeat(1,3,1,1),style_weight=1e6,content_weight=0.1)
			loss_content = content_criteria(y,x)
			loss = 1*(-energy) + closs + sloss + 10*loss_content # for style, content loss
		else:
			loss_content = content_criteria(y,x) # for l1 l2 loss
			loss = 1*(-energy)+10*loss_content # for l1 l2 loss
		loss.backward()
		sgd.step()

	if return_seq:
		return sequence
	else:
		return x.clone().detach(),loss


def neg_langvin_sampler(model, x,y, langevin_steps=20, lr=1.0, sigma=0e-2, return_seq=False,target_slice = None,ae=None,cfg=None,pix=False):
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
  
		if pix:
			# content_criteria = ssim
			recon_latent = x.unsqueeze(1).repeat(1, ae.mapping_fl.num_layers, 1)
			recon_slices = decode(ae, recon_latent, cfg) 
   
			org_latent = x_orig.unsqueeze(1).repeat(1, ae.mapping_fl.num_layers, 1)
			org_slices = decode(ae,org_latent,cfg)

			closs, sloss = calculate_losses(cnn,target_slice.repeat(1,3,1,1),org_slices.repeat(1,3,1,1),recon_slices.repeat(1,3,1,1),style_weight=1e6,content_weight=0.1)
			loss_content = content_criteria(y,x)
			loss = -(1*(-energy) + closs + sloss+ 10*loss_content) # for style, content loss
		else:
			loss_content = content_criteria(y,x) # for l1 l2 loss
			loss = -(1*(-energy)+10*loss_content) # for l1 l2 loss


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

#
def decode(model, x, cfg):
	layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
	ones = torch.ones(layer_idx.shape, dtype=torch.float32)
	coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
	return model.decoder(x, cfg.MODEL.LAYER_COUNT - 1, 1, noise=True)

  

def train(cfg, logger):
	# Create save path
	prefix = cfg.DATA.NAME + "-" + cfg.DATA.SOURCE + '2' + cfg.DATA.TARGET
	save_path = os.path.join("results", prefix)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	suffix = "-".join([item for item in [
		"ls%d" % (cfg.LANGEVIN.STEP),
		# "llr%.2f" % (cfg.LANGEVIN.LR),
		"lr%.4f" % (cfg.EBM.LR),
		"h%d" % (cfg.EBM.HIDDEN),
		"layer%d" % (cfg.EBM.LAYER),
		f"{RUN_NAME}"
		# "opt%s" % (cfg.EBM.OPT),
	] if item is not None])
	run_dir = _create_run_dir_local(save_path, suffix)
	# _copy_dir(['translation'], run_dir)
	sys.stdout = Logger(os.path.join(run_dir, 'log.txt'))
	print(cfg)
	

	ae = load_ae(cfg, logger).to('cuda')

	device = 'cuda'

	data_root = Path(cfg.DATA.ROOT) 
	dataset_root = data_root / cfg.DATASET.ROOT 
	dataset_path = dataset_root / cfg.DATA.NAME 
	print(dataset_path)
 
	label_root = dataset_root / 'train_labels'
	source_label = label_root / f'{cfg.DATA.SOURCE}.tsv'
	target_label = label_root / f'{cfg.DATA.TARGET}.tsv'
	

	source_dataset = MRIDataset_processed(dataset_path,'npy',5,annotation_file=target_label)
	source_sampler = dist.data_sampler(source_dataset, shuffle=True, distributed=False)
	source_loader = DataLoader(source_dataset, batch_size=cfg.DATA.BATCH, sampler=source_sampler, num_workers=0, drop_last=False, generator=torch.Generator(device='cuda'))


	latent_ebm = LatentEBM(latent_dim=512, n_layer=cfg.EBM.LAYER, n_hidden=cfg.EBM.HIDDEN).to(device)

	latent_ema = LatentEBM(latent_dim=512, n_layer=cfg.EBM.LAYER, n_hidden=cfg.EBM.HIDDEN).to(device)
	ema(latent_ema, latent_ebm, decay=0.)
 
	# generate_recon(cfg,ae,latent_ebm,run_dir,0)

	latent_optimizer = optim.SGD(latent_ebm.parameters(), lr=cfg.EBM.LR)
	if cfg.EBM.OPT == 'adam':
		latent_optimizer = optim.Adam(latent_ebm.parameters(), lr=cfg.EBM.LR)

	layer_count = cfg.MODEL.LAYER_COUNT
	iterations = 0
	batch_size = cfg.DATA.BATCH
	fix_sample = True
	# nrow = min(cfg.DATA.BATCH, 12)
	SGLD_PIX = False

	
	if fix_sample:
		source_iter = iter(source_loader)
		sample_target_batch = next(source_iter)
		sample_source_slices = torch.cat((sample_target_batch[:,0,:,:],sample_target_batch[:,5,:,:]),0).unsqueeze(1).float().to(device)
		sample_target_slices = torch.cat((sample_target_batch[:,0,:,:],sample_target_batch[:,5,:,:]),0).unsqueeze(1).float().to(device)
	
	nrow = cfg.DATA.BATCH

	# ## sanity check for paired image
	# import matplotlib.pyplot as plt
	# for slic in range(9):
	# 	src = sample_source_slices.to('cpu').squeeze()[slic]
	# 	tar = sample_target_slices.to('cpu').squeeze()[slic]
	# 	fig, ax = plt.subplots(1,2)
	# 	ax[0].imshow(src,cmap='gray')
	# 	ax[1].imshow(tar,cmap='gray')
	# plt.show()
	# ##
 
	ebm_param = sum(p.numel() for p in latent_ebm.parameters())
	ae_param = sum(p.numel() for p in ae.parameters())
	print(ebm_param, ae_param)
 
	best_model_wgt = copy.deepcopy(latent_ebm.state_dict())
	best_loss = 999
	best_epoch = 0
	best_ssim = 0
	best_psnr = 0
 
	for epoch in trange(0,cfg.EBM.EPOCHS,desc='epoch'):
		running_loss = 0.0
		running_SSIM = 0.0
		running_SSIM_unharm = 0.0
		running_PSNR = 0.0

		for  batch_target in tqdm(source_loader,desc='batches'):

			for slice in range(batch_target.shape[1]):
				# tqdm.write(f'slice: {slice}')
				iterations += 1
				target_slices = batch_target[:,slice,:,:].unsqueeze(1).float()
				target_slices = target_slices.to(device)
				latent_ebm.zero_grad()
				latent_optimizer.zero_grad()

				target_latent = encode(ae, target_slices, cfg)
			    
				# randomize source latent
				rnd = np.random
				source_latent = rnd.randn(cfg.DATA.BATCH, cfg.MODEL.LATENT_SPACE_SIZE)
				source_latent = torch.tensor(source_latent).float().cuda()
				source_latent = ae.mapping_fl(source_latent)[:,0,:] # (9,14,512)
    
				target_latent = target_latent.squeeze()

				requires_grad(latent_ebm, False) 
				source_latent_q,_ = langvin_sampler(latent_ebm, source_latent.clone().detach(),target_latent.clone().detach(),
												langevin_steps=cfg.LANGEVIN.STEP, lr=cfg.LANGEVIN.LR,target_slice=target_slices,ae=ae,cfg=cfg,pix=SGLD_PIX)
	
				######################################## compute recovered source latent for Cycle-consistency
				recovered_source_latent , _ = neg_langvin_sampler(latent_ebm, source_latent_q.clone().detach(),source_latent.clone().detach(),
														langevin_steps=cfg.LANGEVIN.STEP, lr=cfg.LANGEVIN.LR,target_slice=target_slices,ae=ae,cfg=cfg,pix=SGLD_PIX)
				##########################################
	

				# EBM loss
				requires_grad(latent_ebm, True)
				source_energy = latent_ebm(source_latent_q)
				source_energy_original = latent_ebm(source_latent)
				target_energy = latent_ebm(target_latent)
				loss = -(target_energy - source_energy).mean()
	
				######cycle_loss####
				latent_criteria = torch.nn.L1Loss()

				l_cycle = latent_criteria(source_latent, recovered_source_latent)
				###################
	
				# content loss
				content_criteria = torch.nn.L1Loss() # l1 loss
				# content_criteria = ssim

	
				recon_latent = source_latent_q.unsqueeze(1).repeat(1, ae.mapping_fl.num_layers, 1)
				recon = decode(ae, recon_latent, cfg)  # (9,1,256,256)
				loss_content = content_criteria(target_slices,recon) # l1, l2 loss
    
				# SSIM
				ssim_score = 1-ssim_loss(target_slices,recon,window_size=5,max_val=1.0,reduction='mean')

				psnr_score = 1-psnr_loss(target_slices,recon,max_val=1.0)
				loss_ssim = 1-ssim_score
				
				loss_total = loss + 10*l_cycle
	

				if abs(loss.item() > 10000):
					break
				latent_ebm.zero_grad()
				latent_optimizer.zero_grad()
				loss_total.backward()
				latent_optimizer.step()

				running_loss += loss_total.item()*target_slices.shape[0] # loss*batchsize
				running_SSIM += ssim_score*target_slices.shape[0]
				running_PSNR += psnr_score*target_slices.shape[0]

				ema(latent_ema, latent_ebm, decay=0.999)


			if iterations % 100 == 0:
				tqdm.write(f'Iter: {iterations}, EBM Loss: {loss:6.3f}, Loss_content: {loss_content:6.3f}, Loss_cycle: {l_cycle:6.3f}, Total_loss: {loss_total:6.3f}')
				tqdm.write(f'current ssim: {ssim_score:6.4f}')
								
				# if fix_sample:
				# 	# source_slices = sample_source_slices
				# 	source_latent = encode(ae,sample_source_slices,cfg)
				# 	source_latent = source_latent.squeeze()

				# 	target_latent = encode(ae,sample_target_slices,cfg)
				# 	target_latent = target_latent.squeeze()

				# randomize source latent
				rnd = np.random
				source_latent = rnd.randn(cfg.DATA.BATCH, cfg.MODEL.LATENT_SPACE_SIZE)
				source_latent = torch.tensor(source_latent).float().cuda()
				source_latent = ae.mapping_fl(source_latent)[:,0,:] # (9,14,512)
    
				latents,_ = langvin_sampler(latent_ema, source_latent[:nrow].clone().detach(),target_latent[:nrow].clone().detach(),
										langevin_steps=cfg.LANGEVIN.STEP, lr=cfg.LANGEVIN.LR,target_slice=sample_target_slices,ae=ae,cfg=cfg,pix=SGLD_PIX)

				with torch.no_grad():
					latents = torch.cat((source_latent[:nrow], latents)) # original src latent, src_to_tar latent
					latents = latents.unsqueeze(1).repeat(1, ae.mapping_fl.num_layers, 1)

					out = decode(ae, latents, cfg) #source_reconstructed, source_to_target_transformed
					out_transfromed = out[nrow:]

					out = torch.cat((sample_source_slices[:nrow], out), dim=0) # source_original, source_reconstructed, source_to_target_transformed
					# out = torch.cat((out,diff),dim=0)
					out = torch.cat((out,sample_target_slices[:nrow]),dim=0) # source_orig, source_recon, source_to_target, target_orig
					saved_img = utils.save_image(
						out,
						f"{run_dir}/{epoch}_{str(iterations).zfill(6)}.png",
						nrow=nrow,
						normalize=False,
						padding=0,
					)
		
		if abs(loss.item() > 10000):
			break


		if (epoch+1) % 5 ==0:
			epoch_SSIM_unharm = running_SSIM_unharm /(len(source_loader.dataset)*batch_target.shape[1])
			tqdm.write(f'unharmonized SSIM: {epoch_SSIM_unharm:6.4f}')

		# end of epoch statistics
		epoch_loss = running_loss /(len(source_loader.dataset)*batch_target.shape[1])
		epoch_SSIM = running_SSIM /(len(source_loader.dataset)*batch_target.shape[1])
		epoch_PSNR = running_PSNR / (len(source_loader.dataset)*batch_target.shape[1])
		writer.add_scalar('ebm loss',epoch_loss,epoch)
		writer.add_scalar('SSIM score',epoch_SSIM,epoch)
		writer.add_scalar('PSNR score',epoch_PSNR,epoch)

		if epoch_SSIM > best_ssim:
			best_loss = epoch_loss
			best_ssim = epoch_SSIM
			best_epoch = epoch
			best_model_wgt = copy.deepcopy(latent_ema.state_dict())
	
	ebm_final = latent_ema
	model_pt = f'{run_dir}/model_final.pt'
	torch.save({
		'epoch':epoch,
		'model_state_dict':ebm_final.state_dict(),
		'optimizer_state_dict':latent_optimizer.state_dict(),
	},model_pt)
 
	latent_ebm.load_state_dict(best_model_wgt)
	ebm_final = latent_ebm
	model_pt = f'{run_dir}/model_best.pt'
	torch.save({
		'epoch':epoch,
		'model_state_dict':ebm_final.state_dict(),
		'optimizer_state_dict':latent_optimizer.state_dict(),
	},model_pt)
	tqdm.write(f'saved epoch {best_epoch} as the best epoch weight. Best SSIM: {best_ssim}')

if __name__ == "__main__":
	gpu_count = 1
	run(train, get_cfg_defaults(), description='Image-Translation', default_config='configs/OpenBHB_syn.yaml',
		world_size=gpu_count, write_log=False)
