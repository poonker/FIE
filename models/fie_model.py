# -*- coding: UTF-8 -*-
"""
@Function:
@File: fiegan_model.py
@Date: 2024/9/18 19:04 
@Author: funky
"""

import torch
import itertools
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.HFC_filter import HFCFilter
import torch.nn as nn
from util.image_pool import ImagePool

def mul_mask(image, mask):
    return (image + 1) * mask - 1

class FIEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            # parser.set_defaults(pool_size=0, gan_mode='vanilla')
            # DP and HF loss depend on the data, we recommend [0.1, 1]
            parser.add_argument('--lambda_D_HF', type=float, default=1.0, help='weight for DHF loss')
            parser.add_argument('--lambda_D', type=float, default=1.0, help='weight for D loss')
            parser.add_argument('--lambda_G_G', type=float, default=10.0, help='weight for G gan loss')
            parser.add_argument('--lambda_G_D', type=float, default=1, help='weight for G gan loss')
            parser.add_argument('--lambda_G_D_IC', type=float, default=1.0, help='weight for G gan loss')
            parser.add_argument('--lambda_G_H', type=float, default=10.0, help='weight for G gan loss')
            parser.add_argument('--lambda_G_IMIC', type=float, default=1.0, help='weight for G gan loss')
            parser.add_argument('--lambda_G_HFIC', type=float, default=1.0, help='weight for G gan loss')

            parser.add_argument('--RMS', action='store_true',)
        parser.add_argument('--filter_width', type=int, default=53, help='weight for G loss')
        parser.add_argument('--nsig', type=int, default=9, help='weight for G loss')
        parser.add_argument('--sub_low_ratio', type=float, default=1.0, help='weight for L1L loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.input_nc = opt.input_nc

        self.loss_names = ['D', 'D_fake', 'D_real',
                           'D_IC','D_IC_fake','D_HF_real',
                           'G', 'G_G','G_D','G_D_IC','G_H', 'G_IMIC']

        self.visual_names = ['noise_good', 'rec_good','rec_good_H',
                             'real_good','real_good_H','rec_ic'
                                ]
        #high frequency
        self.hfc_filter = HFCFilter(opt.filter_width, nsig=opt.nsig, sub_low_ratio=opt.sub_low_ratio, sub_mask=True, is_clamp=True).to(self.device)

        if self.isTrain:
            self.model_names = ['G', 'D_HF', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
            # self.visual_names = ['noise_good', 'rec_good',
            #                  'real_good','real_good_H','rec_ic']
            self.visual_names = ['rec_good'
                             ]
            
        self.netG = networks.define_G(6, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.gatingnetwork = networks.GatingNetwork(3,16,opt.init_type, opt.init_gain, self.gpu_ids)
        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        if self.isTrain:
            self.netD = networks.define_D(3, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_HF = networks.define_D(3, opt.ndf, opt.netD_HF,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
          
            self.noise_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.real_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images          
          
            # loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionGAN1 = networks.GANLoss(opt.gan_mode1).to(self.device)
            self.criterionGAN2 = networks.GANLoss(opt.gan_mode2).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionH = torch.nn.L1Loss()

            self.criterionL2 = torch.nn.L1Loss()

            # optimizers
            if not self.opt.RMS:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.gatingnetwork.parameters(),
                                                    self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters(),
                                                                    self.netD_HF.parameters()), lr=opt.lr,
                                                    betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.RMSprop(itertools.chain(self.gatingnetwork.parameters(),
                                                    self.netG.parameters()), lr=opt.lr, alpha=0.9)
                self.optimizer_D = torch.optim.RMSprop(itertools.chain(self.netD.parameters(),
                                                                    self.netD_HF.parameters()), lr=opt.lr,
                                                    alpha=0.9)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """
        处理输入
        """
        if self.isTrain:
            #AtoB = self.opt.direction == 'AtoB'
            self.noise_good = input['noise_good'].to(self.device)
            self.real_good = input['real_good'].to(self.device)
            #self.noise_good_mask = input['noise_good_mask'].to(self.device)
            self.real_good_mask = input['real_good_mask'].to(self.device)
            self.real_good_H = self.hfc_filter(self.real_good,self.real_good_mask)
            self.noise_good_H = self.hfc_filter(self.noise_good,self.real_good_mask)

        else:
            #AtoB = self.opt.direction == 'AtoB'
            self.noise_good = input['noise_good'].to(self.device)
            self.real_good = input['real_good'].to(self.device)
            #self.noise_good_mask = input['noise_good_mask'].to(self.device)
            self.real_good_mask = input['real_good_mask'].to(self.device)            
            self.real_good_H = self.hfc_filter(self.real_good,self.real_good_mask)

            self.image_paths = input['noise_good_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            
            self.alpha = self.gatingnetwork(self.real_good)
            self.noiseb = self.noise_A_pool.query(self.noise_good)
            self.inputb = self.alpha*self.noise_good + (1-self.alpha)*self.noiseb
            self.input6 = torch.cat([self.noise_good, self.inputb], dim=1)

            self.rec_good,self.rec_ic = self.netG(self.input6)  
            self.rec_good = mul_mask(self.rec_good, self.real_good_mask)
            self.rec_ic = mul_mask(self.rec_ic, self.real_good_mask)

            self.rec_good_H = self.hfc_filter(self.rec_good, self.real_good_mask)
            self.rec_good_H = mul_mask(self.rec_good_H, self.real_good_mask)
        else:
            self.alpha = self.gatingnetwork(self.real_good_H)
            self.input6 = torch.cat([self.noise_good, self.alpha*self.noise_good_H], dim=1)
            self.rec_good,self.rec_ic = self.netG(self.input6)  
            self.rec_good = mul_mask(self.rec_good, self.real_good_mask)
            self.rec_ic = mul_mask(self.rec_ic, self.real_good_mask)

    def backward_D_HF(self):
        """
        Calculate high frequency loss for the discriminator, we want to closer rec_good'F and real_good'F
        """
        # Fake Target, detach
        pred_rec_ic = self.netD_HF(self.rec_ic.detach())
        pred_real_good_H = self.netD_HF(self.real_good_H.detach())

        self.loss_D_IC_fake = self.criterionGAN2(pred_rec_ic, False) * self.opt.lambda_D_HF
        self.loss_D_HF_real = self.criterionGAN2(pred_real_good_H, True) * self.opt.lambda_D_HF
        self.loss_D_IC = (self.loss_D_IC_fake + self.loss_D_HF_real) * 0.5
        self.loss_D_IC.backward()

    def backward_D(self):
        """
        Calculate GAN loss for the discriminator
        """
        #pred_noise_good = self.netD(self.noise_good.detach())
        pred_rec_good = self.netD(self.rec_good.detach())
        pred_real_good = self.netD(self.real_good.detach())

        self.loss_D_fake = self.criterionGAN1(pred_rec_good, False)* self.opt.lambda_D
        self.loss_D_real = self.criterionGAN1(pred_real_good, True)* self.opt.lambda_D

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        """
        Calculate GAN and L1 loss for the generator
        Generator should fool the D and D_HF
        """
        # First, G(A) should fake the discriminator
        pred_rec_good = self.netD(self.rec_good)
        pred_rec_ic = self.netD_HF(self.rec_ic)

        self.loss_G_D = self.opt.lambda_G_D * self.criterionGAN1(pred_rec_good, True)
        self.loss_G_D_IC = self.opt.lambda_G_D_IC * self.criterionGAN2(pred_rec_ic, True)

        self.loss_G_G = self.criterionL2(self.rec_good, self.real_good) * self.opt.lambda_G_G
        self.loss_G_H = self.criterionH(self.rec_good_H, self.real_good_H) * self.opt.lambda_G_H
        self.loss_G_IMIC = self.criterionL1(self.real_good_H,self.rec_ic) * self.opt.lambda_G_IMIC

        self.loss_G = self.loss_G_G + self.loss_G_H + self.loss_G_D + self.loss_G_D_IC + self.loss_G_IMIC       
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update DD (domain discriminator)
        self.set_requires_grad([self.netD, self.netD_HF], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.backward_D_HF()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        # update G
        self.set_requires_grad([self.netD, self.netD_HF],
                               False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights


