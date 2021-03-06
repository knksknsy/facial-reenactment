from pytorch_msssim import ssim
from dataset.utils import denormalize

def calculate_ssim(images_fake, images_real, normalize):
    ssim_val = ssim(
        denormalize(images_fake.detach().clone(), mean=normalize[0], std=normalize[1]),
        denormalize(images_real.detach().clone(), mean=normalize[0], std=normalize[1]),
        data_range=1.0, size_average=False
    )
    return ssim_val
