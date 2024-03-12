# U-Sketch: An Efficient Approach for Sketch to Image Diffusion Models

This repository contains the implementation of our paper titled *U-Sketch: An Efficient Approach for Sketch to Image Diffusion Models*.

## Abstract
Diffusion models have demonstrated remarkable performance in text-to-image synthesis, producing realistic and high resolution images that faithfully adhere to the corresponding text-prompts. 
 Despite their great success, they still fall behind in sketch-to-image synthesis tasks, where in addition to text-prompts, the spatial layout of 
 the generated images has to closely follow the outlines of certain reference sketches. 
 Employing an MLP latent edge predictor to guide the spatial layout of the synthesized image by predicting edge maps at each denoising step has been 
 recently proposed. Despite yielding promising results, the pixel-wise operation of the MLP does not take into account the spatial layout as a whole, and demands numerous denoising iterations to produce satisfactory 
 images, leading to time inefficiency. To this end, we introduce U-Sketch, a framework featuring a U-Net type latent edge predictor, which is capable of efficiently capturing both local and global features, as well 
 as spatial correlations between pixels. Moreover, we propose the addition of a sketch simplification network that offers the user the choice of preprocessing and simplifying input sketches for enhanced outputs. 
 The experimental results, corroborated by user feedback, demonstrate that our proposed U-Net latent edge predictor leads to more realistic results, that are better aligned with the spatial outlines of the reference sketches, 
 while drastically reducing the number of required denoising steps and, consequently, the overall execution time.

## Installation
To run the code, please follow these steps:
1. Create a new directory named *models-checkpoints*
```
mkdir models-checkpoints
```
2. Download Unet Latent Edge Predictor checkpoint
```
pip install gdown 
```
```
cd models-checkpoints
gdown https://drive.google.com/uc?id=1w7eimErXnnRrjY6TY8yXrmno-hvcZ94_
```
3. Download [Sketch Simplification Network](https://mega.nz/folder/2lUn1YbY#JhTkB1vdaBMeTCSs37iTVA/file/OsdxVQaD) checkpoint into the models-checkpoint directory (model_gan.pth)
4. Try U-Sketch demo in the Jupyter notebook `usketch-demo.ipynb`

## Citations
*Sketch Simplification Network*
-  Simo-Serra, E., Iizuka, S., Sasaki, K., Ishikawa, H.: Learning to simplify: fully
convolutional networks for rough sketch cleanup, 2016.  https://doi.org/10.1145/2897824.2925972
- Simo-Serra, E., Iizuka, S., Ishikawa, H.: Mastering sketching: Adversarial augmentation for structured prediction, 2017. http://arxiv.org/abs/1703.08966
  
*MLP baseline*
- Voynov, Andrey, Kfir Aberman, and Daniel Cohen-Or: Sketch-guided text-to-image diffusion models. ACM SIGGRAPH 2023 Conference Proceedings. 2023.
See implementation [here](https://github.com/Mikubill/sketch2img).
