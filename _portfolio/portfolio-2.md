---
title: "Conditional Variational Autoencoder for facial attribute manipulation"
excerpt: "A conditional variational autoencoder (CVAE) trained on the CelebA dataset to generate facial images conditioned on specific attributes, enabling controlled facial attribute manipulation.<br/><img src='/images/CVAE.png'>"
collection: portfolio
---

A Beta-CVAE for facial attribute manipulation using the CelebA dataset, incorporating a beta-variational objective to balance reconstruction fidelity and latent disentanglement. The model enabled conditional generation of facial images based on user-specified attributes. Its latent space exhibited controlled attribute-specific transformations, facilitating interpretability and targeted image synthesis.
The model was trained on a subset of 10,000 out of 200,000+ images from the CelebA dataset. The training was conducted on Google Colab Pro using an NVIDIA A100 GPU.

<div style="margin-bottom: 20px; text-align: center;">
  <img src="/images/Generated_2.png" alt="Generated" width="1457" height="887">
  <p style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; display: inline-block; margin-top: 10px;">
    A sufficiently disentangled representation while maintaining acceptable reconstruction quality.
  </p>
</div>

<div style="margin-bottom: 20px; text-align: center;">
  <img src="/images/Generated_1.png" alt="Generated" width="1457" height="887">
  <p style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; display: inline-block; margin-top: 10px;">
    Disentanglement was emphasized, enabling more interpretable latent spaces but at the cost of reduced reconstruction quality. Beta = 4
  </p>
</div>

<div style="margin-bottom: 20px; text-align: center;">
  <img src="/images/supervised_gradual_1.png" alt="Generating Progress" width="1457" height="887">
  <p style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; display: inline-block; margin-top: 10px;">
    Generated faces to exhibit gradual addition of facial features.
  </p>
</div>
