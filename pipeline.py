import torch
import torch.nn as nn
import numpy as np
import random 
import math
import tqdm
import gc
import matplotlib.pyplot as plt

from PIL import Image
from itertools import repeat
from torchvision import transforms, utils
from torchvision.transforms import v2
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageStat
from functools import reduce


def hook_unet(unet):
    blocks_idx = [0, 1, 2]
    feature_blocks = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        
        if isinstance(output, torch.TensorType):
            feature = output.float()
            setattr(module, "output", feature)
        elif isinstance(output, dict): 
            feature = output.sample.float()
            setattr(module, "output", feature)
        else: 
            feature = output.float()
            setattr(module, "output", feature)
    
    # 0, 1, 2 -> (ldm-down) 2, 4, 8
    for idx, block in enumerate(unet.down_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block) 
            
    # ldm-mid 0, 1, 2
    for block in unet.mid_block.attentions + unet.mid_block.resnets:
        block.register_forward_hook(hook)
        feature_blocks.append(block) 
    
    # 0, 1, 2 -> (ldm-up) 2, 4, 8
    for idx, block in enumerate(unet.up_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block)  
            
    return feature_blocks
    

class SketchGuidedText2Image():
    
    def __init__(self, stable_diffusion_pipeline, unet, vae, text_encoder, lep_unet, scheduler, tokenizer, sketch_simplifier, device):
        super().__init__()
        self.stable_diffusion_pipeline = stable_diffusion_pipeline
        
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.lep_unet = lep_unet
        self.lep_unet.eval()
        self.lep_unet.requires_grad_(False)
        
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.device = device
        
        self.sketch_simplifier = sketch_simplifier
       
        self.feature_blocks = hook_unet(self.unet)
        
    
    @torch.no_grad()
    def Inference(self, prompt, negative_prompt, num_inference_timesteps, edge_maps, num_images_per_prompt = 1, classifier_guidance_strength = 8, sketch_guidance_strength = 1.8, seed = None, simplify_edge_maps = False, guidance_steps_perc = 0.5):
        if seed:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            
        # Set number of inference steps
        self.scheduler.set_timesteps(num_inference_timesteps)
        
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # Create text embeddings
        final_text_embeddings =  self.stable_diffusion_pipeline._encode_prompt(
            prompt = prompt,
            device = self.device,
            num_images_per_prompt = num_images_per_prompt,
            do_classifier_free_guidance = True)
        
        # Generate random noise and convert it to its latent representation
        noise = torch.randn(batch_size*num_images_per_prompt, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size).to(self.device)

        latents = noise.half() * self.scheduler.init_noise_sigma
        # Prepare edge maps
        if simplify_edge_maps:
            simplified_edge_maps = [self.sketch_simplification(edge_map) for edge_map in edge_maps]
            print("Original Edge Maps - Simplified Edge Maps\n")
            for i, (edge_map, simplified_edge_map) in enumerate(zip(edge_maps, simplified_edge_maps)):
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(np.array(edge_map), cmap = "gray")
                axs[1].imshow(np.array(simplified_edge_map), cmap = "gray")
            plt.show()
            print("Would you like to keep the simplified edge maps? [Y]\[N]\n")
            answer = input()
            if answer == "Y" or answer == "y":
                edge_maps = simplified_edge_maps
                
        encoded_edge_maps = self.prepare_edge_maps(prompt = prompt, num_images_per_prompt = num_images_per_prompt, edge_maps = edge_maps)
        
        gradient = True

        noise = latents.detach().clone()
        for i, timestep in enumerate(tqdm.tqdm(self.scheduler.timesteps)):
            
            gradient_state = torch.enable_grad()
            if i > int(guidance_steps_perc*num_inference_timesteps):
                gradient_state = torch.no_grad()
                gradient = False

            unet_input = self.scheduler.scale_model_input(torch.cat([latents] * 2), timestep).to(self.device)
            unet_input = unet_input.requires_grad_(True)

            with gradient_state:
                u, t = self.unet(unet_input, timestep, encoder_hidden_states = final_text_embeddings).sample.chunk(2)     
            
            pred = u + classifier_guidance_strength*(t-u)
            
            latents_old = unet_input.chunk(2)[1]
            latents = self.scheduler.step(pred, timestep, latents).prev_sample
            
            with gradient_state:        
                    intermediate_result = []
                    for block in self.feature_blocks:
                        outputs = block.output
                        resized = torch.nn.functional.interpolate(outputs, size = latents.shape[2], mode = "bilinear") 
                        intermediate_result.append(resized)
                        del block.output
                    
                    intermediate_result = torch.cat(intermediate_result, dim=1).to(self.device)

                    noise_level = self.get_noise_level(noise, timestep)

                    result = self.lep_unet(intermediate_result, torch.cat([noise_level]*2))
                    _, result = result.chunk(2)

                    if gradient:
                        similarity = torch.linalg.norm(result - encoded_edge_maps)**2

                        _, grad = torch.autograd.grad(similarity,  unet_input)[0].chunk(2)

                        alpha = (torch.linalg.norm(latents_old - latents)/torch.linalg.norm(grad))*sketch_guidance_strength

                        latents -= alpha*grad

            gc.collect()
            torch.cuda.empty_cache()
        
        return {
            "generated_image" : self.latents_to_image(latents),
            "edge_map": self.latents_to_image(result),
        }
    
    @torch.no_grad()
    def sketch_simplification(self, pil_image):
        immean = 0.9664114577640158
        imstd = 0.0858381272736797
        
        data = pil_image.convert('L')
        w, h = data.size[0], data.size[1]
        pw = 8 - (w % 8) if w % 8 != 0 else 0
        ph = 8 - (h % 8) if h % 8 != 0 else 0
        stat = ImageStat.Stat(data)

        data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
        if pw != 0 or ph != 0:
            data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data
        pred = self.sketch_simplifier(data.cuda()).float()
        pred_array = pred.detach().cpu().numpy()[0,0]     
        return Image.fromarray((pred_array*255).astype("uint8")).convert("RGB")

    @torch.no_grad()
    def prepare_edge_maps(self, prompt, num_images_per_prompt, edge_maps):
        batch_size = len(prompt)
        if batch_size != len(edge_maps):
            raise ValueError("Wrong number of edge maps")
            
        processed_edge_maps = [Image.fromarray(255 - np.array(edge_map)[:,:,:3]).resize((512, 512)) for edge_map in edge_maps]
        encoded_edge_maps = [self.image_to_latents(edge_map.resize((512, 512)), img_type = "edge_map") for edge_map in processed_edge_maps]
        encoded_edge_maps_final = [edge_map for item in encoded_edge_maps for edge_map in repeat(item, num_images_per_prompt)]
        encoded_edge_maps_tensor = torch.cat(encoded_edge_maps_final, dim = 0)
        return encoded_edge_maps_tensor

    @torch.no_grad()
    def text_to_embeddings(self, text):

        tokenized_text = self.tokenizer(text, padding = "max_length", max_length = self.tokenizer.model_max_length, truncation = True, return_tensors = "pt")

        with torch.no_grad(): 
            text_embeddings = text_encoder(
                tokenized_text.input_ids.to(self.device)
            )[0].half()
        return text_embeddings
    
    def get_noise_level(self, noise, timesteps):
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        
        while len(sqrt_one_minus_alpha_prod.shape) < len(noise.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noise_level = sqrt_one_minus_alpha_prod.to(self.device) * noise
        return noise_level

    @torch.no_grad()
    def image_to_latents(self, img, img_type):
      np_img = (np.array(img).astype(np.float32) / 255.0) 
      if img_type == "edge_map":
            np_img[np_img < 0.5] = 0.
            np_img[np_img >= 0.5] = 1.
      np_img = np_img* 2.0 - 1.0
      np_img = np_img[None].transpose(0, 3, 1, 2)
      torch_img = torch.from_numpy(np_img)
      generator = torch.Generator(self.device).manual_seed(0)
      latents = self.vae.encode(torch_img.to( self.vae.dtype).to(self.device)).latent_dist.sample(generator=generator)
      latents = latents * 0.18215
      return latents

    @torch.no_grad()
    def latents_to_image(self, latents):
        '''
        Function to convert latents to images
        '''
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

