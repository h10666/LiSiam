'''
Function:
    builder for different models
Author:
    Zhenchao Jin
'''
# from .fcn import FCN
# from .ce2p import CE2P
# from .ccnet import CCNet
# from .danet import DANet
# from .gcnet import GCNet
# from .dmnet import DMNet
# from .encnet import ENCNet
# from .apcnet import APCNet
# from .emanet import EMANet
# from .pspnet import PSPNet
# from .psanet import PSANet
from .ocrnet import *
# from .dnlnet import DNLNet
# from .annnet import ANNNet
# from .upernet import UPerNet
# from .deeplabv3 import Deeplabv3
# from .lrasppnet import LRASPPNet
# from .semanticfpn import SemanticFPN
# from .nonlocalnet import NonLocalNet
# from .deeplabv3plus import Deeplabv3Plus


'''build model'''
def BuildModel(cfg, mode, **kwargs):
    supported_models = {
        # 'LiSiam': LiSiam,
        # 'LiSiam_AS1': LiSiam_AS1,
        # 'LiSiam_AS2': LiSiam_AS2,
        # 'LiSiam_AS3': LiSiam_AS3,
        # 'LiSiam_AS4': LiSiam_AS4,
        # 'LiSiam_AS5': LiSiam_AS5,      
        # 'LiSiam_AS6': LiSiam_AS6,      
        # 'LiSiam_AS7': LiSiam_AS7,      
        # 'LiSiam_AS9': LiSiam_AS9,      
        'LiSiam_A': LiSiam_A,
        'LiSiam_B': LiSiam_B,
        'LiSiam_C': LiSiam_C,
        'LiSiam_D': LiSiam_D,
        'LiSiam_E': LiSiam_E,
        'LiSiam_F': LiSiam_F,        
        'TNet_fa': TNet_fa,    
        'TNet_ff': TNet_ff,    
            
        # TNet_fa  
        # 'SiamV1': SiamV1,
        # 'SiamV2': SiamV2,
        # # 'SiamV3': SiamV3,
        # 'Siam8V0': Siam8V0,
        # 'Siam9V0': Siam9V0,
        # 'Siam9V2': Siam9V2,
        # 'Siam9V3': Siam9V3,
        # 'Siam9V4': Siam9V4,
        # 'Siam9V4_1': Siam9V4_1,
        'Siam9V4_1_inv': Siam9V4_1_inv,
        # 'ocrnetv4': OCRNetV4,
    }
    model_type = cfg['type']
    assert model_type in supported_models, 'unsupport model_type %s...' % model_type
    # print('##loaded model:',model_type)
    return supported_models[model_type](cfg, mode=mode)
    # return model_type(cfg, mode=mode)
    # return models.__dict__[args.arch]

    # supported_models = {
    #     'fcn': FCN,
    #     'ce2p': CE2P,
    #     'ccnet': CCNet,
    #     'danet': DANet,
    #     'gcnet': GCNet,
    #     'dmnet': DMNet,
    #     'encnet': ENCNet,
    #     'apcnet': APCNet,
    #     'emanet': EMANet,
    #     'pspnet': PSPNet,
    #     'psanet': PSANet,
    #     'ocrnet': OCRNet,
    #     'dnlnet': DNLNet,
    #     'annnet': ANNNet,
    #     'upernet': UPerNet,
    #     'deeplabv3': Deeplabv3,
    #     'lrasppnet': LRASPPNet,
    #     'semanticfpn': SemanticFPN,
    #     'nonlocalnet': NonLocalNet,
    #     'deeplabv3plus': Deeplabv3Plus,
    # }