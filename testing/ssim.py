from pytorch_msssim import ssim
from dataset.utils import denormalize

def calculate_ssim(images_fake, images_real):
    ssim_val = ssim(
        denormalize(images_fake.detach().clone(), mean=0.5, std=0.5),
        denormalize(images_real.detach().clone(), mean=0.5, std=0.5),
        data_range=1.0, size_average=False
    )
    return ssim_val
