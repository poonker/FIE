import pywt
import numpy as np
import torch
#import matplotlib.pyplot as plt

# 读取彩色图像
# 你可以替换这里的"your_image_path.jpg"为你的图片路径
# 注意：需要安装PIL或OpenCV库来读取图片
import cv2
#image = cv2.imread("0001.jpg", cv2.IMREAD_COLOR)
def hlfc(images):
    l = []
    h = []
    for i in range(len(images)):
        image = images[i]
        image = image.permute(1,2,0)
        # 将图像转换为RGB格式
        image = image.numpy()
        #image = image.permuate()
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image,(512,512))

        # 分离R、G、B通道 其实是bgr吧
        r_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        b_channel = image[:, :, 2]

        # 对每个通道进行小波变换
        coeffs_r = pywt.dwt2(r_channel, 'haar')
        coeffs_g = pywt.dwt2(g_channel, 'haar')
        coeffs_b = pywt.dwt2(b_channel, 'haar')


        # 获取每个通道的低频和高频分量
        cA_r, (cH_r, cV_r, cD_r) = coeffs_r
        cA_g, (cH_g, cV_g, cD_g) = coeffs_g
        cA_b, (cH_b, cV_b, cD_b) = coeffs_b

        # 重构每个通道的低频分量（近似分量）
        reconstructed_r = pywt.idwt2((cA_r, (None, None, None)), 'haar')
        reconstructed_g = pywt.idwt2((cA_g, (None, None, None)), 'haar')
        reconstructed_b = pywt.idwt2((cA_b, (None, None, None)), 'haar')

        reconstructed_hr = pywt.idwt2((cH_r, (None, None, None)), 'haar')
        reconstructed_hg = pywt.idwt2((cH_g, (None, None, None)), 'haar')
        reconstructed_hb = pywt.idwt2((cH_b, (None, None, None)), 'haar')

        reconstructed_vr = pywt.idwt2((cV_r, (None, None, None)), 'haar')
        reconstructed_vg = pywt.idwt2((cV_g, (None, None, None)), 'haar')
        reconstructed_vb = pywt.idwt2((cV_b, (None, None, None)), 'haar')

        reconstructed_dr = pywt.idwt2((cD_r, (None, None, None)), 'haar')
        reconstructed_dg = pywt.idwt2((cD_g, (None, None, None)), 'haar')
        reconstructed_db = pywt.idwt2((cD_b, (None, None, None)), 'haar')

        # 合并R、G、B通道得到重构的彩色图像
        reconstructed_image = np.zeros_like(image)
        reconstructed_image[:, :, 0] = reconstructed_r
        reconstructed_image[:, :, 1] = reconstructed_g
        reconstructed_image[:, :, 2] = reconstructed_b

        reconstructed_himage = np.zeros_like(image)
        reconstructed_himage[:, :, 0] = reconstructed_hr
        reconstructed_himage[:, :, 1] = reconstructed_hg
        reconstructed_himage[:, :, 2] = reconstructed_hb

        reconstructed_vimage = np.zeros_like(image)
        reconstructed_vimage[:, :, 0] = reconstructed_vr
        reconstructed_vimage[:, :, 1] = reconstructed_vg
        reconstructed_vimage[:, :, 2] = reconstructed_vb

        reconstructed_dimage = np.zeros_like(image)
        reconstructed_dimage[:, :, 0] = reconstructed_dr
        reconstructed_dimage[:, :, 1] = reconstructed_dg
        reconstructed_dimage[:, :, 2] = reconstructed_db

        # # 合并R、G、B通道得到重构的彩色图像
        # #cA_r, (cH_r, cV_r, cD_r)
        # reconstructed_image = np.zeros((image.shape[0]//2,image.shape[1]//2,image.shape[2]))
        # reconstructed_image[:, :, 0] = cA_r
        # reconstructed_image[:, :, 1] = cA_g
        # reconstructed_image[:, :, 2] = cA_b

        # reconstructed_himage = np.zeros((image.shape[0]//2,image.shape[1]//2,image.shape[2]))
        # reconstructed_himage[:, :, 0] = cH_r
        # reconstructed_himage[:, :, 1] = cH_g
        # reconstructed_himage[:, :, 2] = cH_b

        # reconstructed_vimage = np.zeros((image.shape[0]//2,image.shape[1]//2,image.shape[2]))
        # reconstructed_vimage[:, :, 0] = cV_r
        # reconstructed_vimage[:, :, 1] = cV_g
        # reconstructed_vimage[:, :, 2] = cV_b

        # reconstructed_dimage = np.zeros((image.shape[0]//2,image.shape[1]//2,image.shape[2]))
        # reconstructed_dimage[:, :, 0] = cD_r
        # reconstructed_dimage[:, :, 1] = cD_g
        # reconstructed_dimage[:, :, 2] = cD_b

        #re_image = reconstructed_image+reconstructed_himage+reconstructed_vimage+reconstructed_dimage
        h_image = reconstructed_himage+reconstructed_vimage+reconstructed_dimage
        #reconstructed_image = torch.from_numpy(reconstructed_image)
        #h_image = torch.from_numpy(h_image)
        reconstructed_image = reconstructed_image.transpose(2,0,1)
        h_image = h_image.transpose(2,0,1)
        l.append(reconstructed_image)
        h.append(h_image)
    l = torch.tensor(l).to(torch.float32)
    h = torch.tensor(h).to(torch.float32)
    return l,h

# 可视化原始彩色图像和重构后的彩色图像
# plt.figure(figsize=(8, 8))
# plt.subplot(4, 1, 1)
# plt.imshow(image)
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(4, 1, 2)
# plt.imshow(reconstructed_image.astype(np.uint8))
# plt.title('Reconstructed Image')
# plt.axis('off')

# plt.subplot(4, 1, 3)
# plt.imshow(reconstructed_dimage.astype(np.uint8))
# plt.title('Reconstructedd Image')
# plt.axis('off')

# plt.subplot(4, 1, 4)
# plt.imshow(re_image.astype(np.uint8))
# plt.title('Re Image')
# plt.axis('off')

# plt.tight_layout()
# plt.show()