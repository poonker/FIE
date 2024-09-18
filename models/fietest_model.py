from .base_model import BaseModel
from . import networks
import torch
from models.guided_filter_pytorch.HFC_filter import HFCFilter

class FIETestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        #parser.set_defaults(dataset_mode='single')
        parser.set_defaults(dataset_mode='test')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        #parser.add_argument('--model_suffix', type=str, default='_A', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        parser.add_argument('--filter_width', type=int, default=53, help='weight for G loss')
        parser.add_argument('--nsig', type=int, default=9, help='weight for G loss')
        parser.add_argument('--sub_low_ratio', type=float, default=1.0, help='weight for L1L loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        #self.visual_names = ['real', 'fake']
        self.visual_names = ['fake']
        self.hfc_filter = HFCFilter(opt.filter_width, nsig=opt.nsig, sub_low_ratio=opt.sub_low_ratio, sub_mask=True, is_clamp=True).to(self.device)

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(6, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.gatingnetwork = networks.GatingNetwork(3,16,opt.init_type, opt.init_gain, self.gpu_ids)
        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['A'].to(self.device)
        self.real_mask = input['A_mask'].to(self.device)
        self.image_paths = input['A_paths']

        self.noise_good_H = self.hfc_filter(self.real,self.real_mask)

        
    def forward(self):
        """Run forward pass."""
        self.alpha = self.gatingnetwork(self.real)
        self.inputb = self.alpha*self.real + (1-self.alpha)*self.real
        self.input6 = torch.cat([self.real, self.inputb], dim=1)        
        self.fake,_ = self.netG(self.input6)  # G(real)
        self.fake = mul_mask(self.fake,self.real_mask) 
    def optimize_parameters(self):
        """No optimization for test model."""
        pass

def mul_mask(image, mask):
    return (image + 1) * mask - 1