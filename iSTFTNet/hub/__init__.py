CKPT_URLS = {
}
import torch
from ..model.generators.istft_generator import Generator

def istft_48k(
    pretrained: bool = True,
    progress: bool = True,
) -> Generator:
    hifigan = Generator(
        initial_channel=128,
        resblock="1",
        resblock_kernel_sizes=[3,7,11],
        resblock_dilation_sizes=[
            [1,3,5],
            [1,3,5],
            [1,3,5]
        ],
        upsample_rates=[8,8,2,2,2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16,16,4,4,4]
        )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            CKPT_URLS["istft-48k"], progress=progress
        )
        hifigan.load_state_dict(checkpoint)
        hifigan.eval()
    return hifigan

def istft_16k(
    pretrained: bool = True,
    progress: bool = True,
) -> Generator:
    hifigan = Generator(
        initial_channel=128,
        resblock="1",
        resblock_kernel_sizes=[3,7,11],
        resblock_dilation_sizes=[
            [1,3,5],
            [1,3,5],
            [1,3,5]
        ],
        upsample_rates=[8,8,2,2,2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16,16,4,4,4]
        )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            CKPT_URLS["istft-16k"], progress=progress
        )
        hifigan.load_state_dict(checkpoint)
        hifigan.eval()
    return hifigan