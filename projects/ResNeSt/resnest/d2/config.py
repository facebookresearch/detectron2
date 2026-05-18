from detectron2.config import CfgNode as CN

def add_resnest_config(cfg):
    """Add config for ResNeSt
    """
    # Place the stride 2 conv on the 1x1 filter
    # Use True only for the original MSRA ResNet;
    # use False for C2 and Torch models
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    # Apply deep stem
    cfg.MODEL.RESNETS.DEEP_STEM = True
    # Apply avg after conv2 in the BottleBlock
    # When AVD=True, the STRIDE_IN_1X1 should be False
    cfg.MODEL.RESNETS.AVD = True
    # Apply avg_down to the downsampling layer for residual path
    cfg.MODEL.RESNETS.AVG_DOWN = True
    # Radix in ResNeSt
    cfg.MODEL.RESNETS.RADIX = 2
    # Bottleneck_width in ResNeSt
    cfg.MODEL.RESNETS.BOTTLENECK_WIDTH = 64
