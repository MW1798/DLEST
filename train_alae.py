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
# - Retained the main functionality of ALAE.
# - Removed codes related to size-growing training.
# - Added an appropriate pixel loss function to ALAE.
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




import torch.utils.data
import torchvision.utils
from torchvision.utils import save_image
from net import *
import os
import utils
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from dataloader import *
import torchvision as tv, torchvision.transforms as tr

from tracker import LossTracker
from model import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm



RUN_NAME = 'ALAE_100perceptual'
writer=SummaryWriter(f'runs/{RUN_NAME}')


def save_sample(lod2batch, tracker, sample, samplez, logger, model, cfg, encoder_optimizer, decoder_optimizer):
	os.makedirs('results', exist_ok=True)

	logger.info('\n[%d/%d] - ptime: %.2f, %s, blend: %.3f, lr: %.12f,  %.12f, max mem: %f",' % (
		(lod2batch.current_epoch + 1), cfg.TRAIN.TRAIN_EPOCHS, lod2batch.per_epoch_ptime, str(tracker),
		lod2batch.get_blend_factor(),
		encoder_optimizer.param_groups[0]['lr'], decoder_optimizer.param_groups[0]['lr'],
		torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

	with torch.no_grad():
		model.eval()
		sample = sample[:lod2batch.get_per_GPU_batch_size()]
		samplez = samplez[:lod2batch.get_per_GPU_batch_size()]

		needed_resolution = model.decoder.layer_to_resolution[lod2batch.lod]
		sample_in = sample
		while sample_in.shape[2] > needed_resolution:
			sample_in = F.avg_pool2d(sample_in, 2, 2)
		assert sample_in.shape[2] == needed_resolution

		blend_factor = lod2batch.get_blend_factor()
		if lod2batch.in_transition:
			needed_resolution_prev = model.decoder.layer_to_resolution[lod2batch.lod - 1]
			sample_in_prev = F.avg_pool2d(sample_in, 2, 2)
			sample_in_prev_2x = F.interpolate(sample_in_prev, needed_resolution)
			sample_in = sample_in * blend_factor + sample_in_prev_2x * (1.0 - blend_factor)

		Z, _ = model.encode(sample_in, lod2batch.lod, blend_factor)

		if cfg.MODEL.Z_REGRESSION:
			Z = model.mapping_fl(Z[:, 0])
		else:
			Z = Z.repeat(1, model.mapping_fl.num_layers, 1)

		rec1 = model.decoder(Z, lod2batch.lod, blend_factor, noise=False)
		rec2 = model.decoder(Z, lod2batch.lod, blend_factor, noise=True)

		Z = model.mapping_fl(samplez)
		g_rec = model.decoder(Z, lod2batch.lod, blend_factor, noise=True) # generated fake image
		dif = abs(sample_in-rec2)


		resultsample = torch.cat([sample_in, rec1, rec2, dif,g_rec], dim=0)

		# @utils.async_func
		def save_pic(x_rec):
			current_epoch = round(lod2batch.current_epoch + lod2batch.iteration * 1.0 / lod2batch.get_dataset_size())
			tracker.register_means(current_epoch)

			result_sample = x_rec
			f = os.path.join(cfg.OUTPUT_DIR,
							 'sample_%d_%d.jpg' % (
								 current_epoch,
								 lod2batch.iteration // 1000)
							 )
			tqdm.write("Saved to %s" % f)
			save_image(result_sample, f, nrow=min(32, lod2batch.get_per_GPU_batch_size()), normalize=True,
					   range=(-1., 1.))

		save_pic(resultsample)


def train(cfg, logger, local_rank, world_size, distributed):

	torch.cuda.set_device(local_rank)
	model = Model(
		startf=cfg.MODEL.START_CHANNEL_COUNT,
		layer_count=cfg.MODEL.LAYER_COUNT,
		maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
		latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
		dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
		style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
		mapping_layers=cfg.MODEL.MAPPING_LAYERS,
		channels=cfg.MODEL.CHANNELS,
		generator=cfg.MODEL.GENERATOR,
		encoder=cfg.MODEL.ENCODER,
		z_regression=cfg.MODEL.Z_REGRESSION
	)
	model.cuda(local_rank)
	model.train()

	if local_rank == 0:
		model_s = Model(
			startf=cfg.MODEL.START_CHANNEL_COUNT,
			layer_count=cfg.MODEL.LAYER_COUNT,
			maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
			latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
			truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
			truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
			mapping_layers=cfg.MODEL.MAPPING_LAYERS,
			channels=cfg.MODEL.CHANNELS,
			generator=cfg.MODEL.GENERATOR,
			encoder=cfg.MODEL.ENCODER,
			z_regression=cfg.MODEL.Z_REGRESSION)
		model_s.cuda(local_rank)
		model_s.eval()
		model_s.requires_grad_(False)

	if distributed:
		model = nn.parallel.DistributedDataParallel(
			model,
			device_ids=[local_rank],
			broadcast_buffers=False,
			bucket_cap_mb=25,
			find_unused_parameters=True)
		model.device_ids = None

		decoder = model.module.decoder
		encoder = model.module.encoder
		mapping_tl = model.module.mapping_tl # mapping_d
		mapping_fl = model.module.mapping_fl # mapping_f
		dlatent_avg = model.module.dlatent_avg
	else:
		decoder = model.decoder  # G
		encoder = model.encoder  # E
		mapping_tl = model.mapping_tl  # mapping_D
		mapping_fl = model.mapping_fl  # mapping_F
		dlatent_avg = model.dlatent_avg


	decoder_optimizer = LREQAdam([
		{'params': decoder.parameters()}, #G
		{'params': mapping_fl.parameters()} #F
	], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

	encoder_optimizer = LREQAdam([
		{'params': encoder.parameters()}, #E
		{'params': mapping_tl.parameters()}, #D
	], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

	scheduler = ComboMultiStepLR(optimizers=
								 {
									'encoder_optimizer': encoder_optimizer,
									'decoder_optimizer': decoder_optimizer
								 },
								 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
								 gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
								 reference_batch_size=32, base_lr=cfg.TRAIN.LEARNING_RATES)

	model_dict = {
		'discriminator': encoder,
		'generator': decoder,
		'mapping_tl': mapping_tl,
		'mapping_fl': mapping_fl,
		'dlatent_avg': dlatent_avg
	}
	if local_rank == 0:
		model_dict['discriminator_s'] = model_s.encoder
		model_dict['generator_s'] = model_s.decoder
		model_dict['mapping_tl_s'] = model_s.mapping_tl
		model_dict['mapping_fl_s'] = model_s.mapping_fl

	tracker = LossTracker(cfg.OUTPUT_DIR)

	checkpointer = Checkpointer(cfg,
								model_dict,
								{
									'encoder_optimizer': encoder_optimizer,
									'decoder_optimizer': decoder_optimizer,
									'scheduler': scheduler,
									'tracker': tracker
								},
								logger=logger,
								save=local_rank == 0)

	extra_checkpoint_data = checkpointer.load()
	logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))


	layer_to_resolution = decoder.layer_to_resolution

	dataset = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS)

	rnd = np.random.RandomState(3456)
	latents = rnd.randn(32, cfg.MODEL.LATENT_SPACE_SIZE)
	samplez = torch.tensor(latents).float().cuda()

	lod2batch = lod_driver.LODDriver(cfg, logger, world_size, dataset_size=len(dataset) * world_size)
					
	if cfg.DATASET.SAMPLES_PATH: # not used 
		path = cfg.DATASET.SAMPLES_PATH

	else:
		dataset.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, lod2batch.get_per_GPU_batch_size())
		sample_loader = make_dataloader(cfg, logger, dataset, lod2batch.get_per_GPU_batch_size(), local_rank)
		sample,lb = next(sample_loader)
  
		sample = sample.to(torch.device(local_rank))


	lod2batch.set_epoch(scheduler.start_epoch(), [encoder_optimizer, decoder_optimizer]) ### initialize? 
 
	# sanity check for save_sample function
	# x = 0
	# save_sample(lod2batch, tracker, sample[:,0,:,:].unsqueeze(1).float(), samplez, x, logger, model_s, cfg, encoder_optimizer,
	# 								decoder_optimizer)
	best_ssim = 0.0	
 

 
	running_loss_g = 0.0
	running_loss_d = 0.0
	running_loss_ae = 0.0
	running_loss_pixel = 0.0
	min_total_loss = 0.0
	with logging_redirect_tqdm():
		per_epoch_ptime = 0
		for epoch in trange(scheduler.start_epoch(), cfg.TRAIN.TRAIN_EPOCHS,desc='epoch'):
			running_SSIM = 0.0
			model.train()
			lod2batch.set_epoch(epoch, [encoder_optimizer, decoder_optimizer]) 
			logger.info("Epoch: %d, Batch size: %d, Batch size per GPU: %d, LOD: %d - %dx%d, blend: %.3f, dataset size: %d" % (
																	epoch,
																	lod2batch.get_batch_size(), 
																	lod2batch.get_per_GPU_batch_size(), 
																	lod2batch.lod,
																	2 ** lod2batch.get_lod_power2(), 
																	2 ** lod2batch.get_lod_power2(),
																	lod2batch.get_blend_factor(),
																	len(dataset) * world_size))

			scheduler.set_batch_size(lod2batch.get_batch_size(), lod2batch.lod)
			dataset.reset(lod2batch.get_lod_power2(), lod2batch.get_per_GPU_batch_size()) 
			batches = make_dataloader(cfg, logger, dataset, lod2batch.get_per_GPU_batch_size(), local_rank) # prepare batches
			model.train()

			need_permute = False
			epoch_start_time = time.time()

			i = 0 # number of slices trained
			j = 0 # number of batches trained
			
			for batch,labels in tqdm(batches,desc='batch'): 
				j +=1
				for slic in trange(batch.shape[1],desc='slice'):
					x_orig = batch[:,slic,:,:].unsqueeze(1).float() 

					i += 1

					x_orig = x_orig.to(torch.device(local_rank))
					x_orig.requires_grad_(True)
					batch_shape = x_orig.shape[0]




					blend_factor = lod2batch.get_blend_factor()

					# update E,D
					encoder_optimizer.zero_grad()
					loss_d = model(x_orig, lod2batch.lod, blend_factor, d_train=True, ae=False)
					tracker.update(dict(loss_d=loss_d))
					loss_d.backward()	
					encoder_optimizer.step()


					# update generator F,G
					decoder_optimizer.zero_grad()
					loss_g = model(x_orig, lod2batch.lod, blend_factor, d_train=False, ae=False)
					tracker.update(dict(loss_g=loss_g))
					loss_g.backward()
					decoder_optimizer.step()

					# update lae E,G
					encoder_optimizer.zero_grad()
					decoder_optimizer.zero_grad()
					lae = model(x_orig, lod2batch.lod, blend_factor, d_train=True, ae=True)
					tracker.update(dict(lae=lae))
					# update E,G pixel loss
					pixel_loss, ssim_loss = model(x_orig, lod2batch.lod, blend_factor, d_train=False, ae=False,pix=True)
					tracker.update(dict(loss_pix=pixel_loss)) ###
					tracker.update(dict(loss_ssim=ssim_loss))

					(lae+100*pixel_loss).backward()
					encoder_optimizer.step()
					decoder_optimizer.step()


					betta = 0.5 ** (lod2batch.get_batch_size() / (10 * 1000.0))
					model_s.lerp(model, betta)
					# tqdm.write('after lerp')

					
					running_loss_g += loss_g.item()*batch_shape
					# tqdm.write('after loss_g')
					running_loss_d += loss_d.item()*batch_shape
					# tqdm.write('after loss_d')
					running_loss_ae += lae.item()*batch_shape
					# tqdm.write('after loss_ae')
					running_loss_pixel += pixel_loss.item()*batch_shape
					# tqdm.write(f'end of {i}')
					running_SSIM += (1-ssim_loss.item())*batch_shape
				# lod_for_saving_model = lod2batch.lod
				lod2batch.step()
	
				if i % 100 == 0: # every 10 batch, since 1 batch has 10 slices
					# write image samples to tensorboard
					# img_grid = torchvision.utils.make_grid(sample_img_batch)
					# writer.add_image(f'sample images from batch {j}, slice {i}',img_grid)
					writer.add_scalar('training loss d',running_loss_d / (batch_shape*batch.shape[1]*10),epoch * len(batches)*batch.shape[1] + i)
					writer.add_scalar('training loss g',running_loss_g / (batch_shape*batch.shape[1]*10),epoch * len(batches)*batch.shape[1] + i)
					writer.add_scalar('training loss ae',running_loss_ae / (batch_shape*batch.shape[1]*10),epoch * len(batches)*batch.shape[1] + i)
					writer.add_scalar('training loss pixel',running_loss_pixel / (batch_shape*batch.shape[1]*10),epoch * len(batches)*batch.shape[1] + i)
					running_loss_d = 0.0
					running_loss_g = 0.0
					running_loss_ae = 0.0
					running_loss_pixel = 0.0

			epoch_end_time = time.time()
			per_epoch_ptime += epoch_end_time - epoch_start_time
			epoch_SSIM = running_SSIM / (len(batches)*batch.shape[0]*batch.shape[1])
			writer.add_scalar('Epoch SSIM',epoch_SSIM,epoch)
			current_total_loss = tracker.tracks['loss_d'].mean()+tracker.tracks['loss_g'].mean()+tracker.tracks['lae'].mean()
			scheduler.step()
			

			if epoch_SSIM > best_ssim:

				best_ssim = epoch_SSIM
				checkpointer.save(f"model_tmp_opt_{epoch+1}")
			save_sample(lod2batch, tracker, sample[:,0,:,:].unsqueeze(1).float(), samplez, logger, model_s, cfg, encoder_optimizer, decoder_optimizer)

		logger.info(f"Training finish!... save training results. Average per epoch time: {per_epoch_ptime/cfg.TRAIN.TRAIN_EPOCHS}")
		if local_rank == 0:
			checkpointer.save("model_final").wait()


if __name__ == "__main__":
	gpu_count = torch.cuda.device_count()
	run(train, get_cfg_defaults(), description='StyleGAN', default_config='ALAE/configs/OpenBHB_1.yaml',
		world_size=gpu_count)
