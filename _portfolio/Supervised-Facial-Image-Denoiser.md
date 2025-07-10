---
title: "Supervised Facial Image Denoiser"
excerpt: "A supervised autoencoder-based image denoiser using the CelebA dataset, achieving effective noise reduction while preserving facial details.<br/><img src='/images/500x300.png'>"
collection: portfolio
---

<img src="/images/denoising_progress.gif" alt="Denoising Progress" width="300" height="300">

A supervised autoencoder for denoising facial images, leveraging the CelebA dataset. Unlike conventional self-supervised denoisers, this approach explicitly maps noisy facial inputs to their clean counterparts, optimizing perceptual image quality using the Structural Similarity Index Measure. Training is conducted with Gaussian noise perturbations, ensuring robustness to real-world noise distributions. The model achieves an SSIM validation loss of 0.2409 using only 10,000 images, demonstrating significant potential for scaling. However, its robustness can be substantially enhanced with the full dataset (200,000 images), enabling better generalization and fidelity.

<img src="/images/Denoise_Result.png" alt="Denoising Progress" width="1457" height="887">

<img src="/images/Denoise_Result_Graph.png" alt="Denoising Progress" width="1457" height="887">