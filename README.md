# FIE - Fundus Image Enhancement

This repository provides the implementation of **A Hybrid CNN-Mamba Model for Multi-Scale Fundus Image Enhancement**, which has been accepted by *Biomedical Optics Express*.

---
![](https://github.com/poonker/FIE/blob/main/image.png)
---

## 🔑 Key Idea
The **Mamba discriminator** may be more efficient than convolutional kernels when treating images as sequential data, similar to processing long sequences of text.

---

## 🚀 How to Use
We recommend familiarity with the **[PyTorch CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)** code paradigm before using this project.

### 📂 Dataset
Following **[Cofenet](https://github.com/joanshen0508/Fundus-correction-cofe-Net)**, we degraded the **[EyeQ](https://github.com/HzFu/EyeQ) dataset** to create paired training data.  
The degradation code is stored in the `tools` folder.  
This paper also integrates **cataract-like degradation** methods.  
You can compile these scripts to generate degraded images that closely resemble real-world scenarios.

---

## 📌 Pretrained Models
We provide pretrained models for easy usage:  
- **256×256** → [Download](https://drive.google.com/file/d/1m6BDQHMupZQFbKgaIHc8-QWmpZRmMmdN/view?usp=drive_link)  
- **512×512** → [Download](https://drive.google.com/drive/folders/17YJdjkjtyluirIaCFiSxejcw2SdQrbH9?usp=drive_link)  
- **1024×1024** → [Download](https://drive.google.com/file/d/1GQ9eXnl9eeS8HlcmKCeAKSrRYm9mzy6y/view?usp=drive_link)  

---

## 🏋️‍♂️ Training (1024×1024)
Run the following command to train the model on **1024×1024** resolution images:
```bash
!python train.py --dataroot ./datasets/eyeq_reference --name 0831_0 --model fie --netG unetd2 \
--netD mambass2 --netD_HF pixel --dataset_mode degraded_with_mask \
--norm instance --gpu_ids 0 --batch_size 1 --lr_policy linear --display_id 1 \
--n_epochs 150 --n_epochs_decay 50 --load_size 1072 --save_epoch_freq 40 --crop_size 1024 \
--PTWH 1 --lr 1e-3 --display_port 8097 --display_env 0831_0 --lambda_G_G 10
```
---

## 🏋️‍♂️ testing (1024×1024)
Run the following command to test the model on **1024×1024** resolution images:
```bash
!python test.py \
--dataroot ./datasets/eyeq_reference \
--name 0831_0 --model fietest --dataset_mode pctest --phase test\
--netG unetd2 --netD_HF pixel --netD mambass2 --norm instance --load_size 1024 --crop_size 1024\
--gpu_ids 0 --batch_size 4 --no_dropout 
```
