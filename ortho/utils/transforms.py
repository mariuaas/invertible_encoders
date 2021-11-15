from .. import (
    torch, np, Image, ImageFilter, gaussian, Tuple, Union
)

_noise_STD = 0.2/1.96

# NOTE: Stupid server has an old version of numpy without typing extensions, so all
#       references to ArrayLike has been removed.
# TODO: Can be changed to classes which return a torchvision.transform.Lambda object?

class SplitResize:

    interpolation_modes = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'hamming': Image.HAMMING,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
    }

    def __init__(
        self, size: Union[int, Tuple[int]], interpolation : str = 'nearest',
        noise_std: float = None, axis: int = 0, upscale: bool = False
    ) -> None:
        assert interpolation in self.interpolation_modes, \
            f'Interpolation mode {interpolation} is not valid.'
        self.size = size
        self.interpolation = self.interpolation_modes[interpolation]
        self.noise_std = noise_std
        self.axis = axis
        self.upscale = upscale

        if isinstance(size, int):
            self.width = self.height = size

        elif len(size) == 1:
            self.width = self.height = size[0]

        else:
            self.height = size[0]
            self.width = size[1]

    def add_noise(self, img: Image):
        img = np.array(img)
        if self.noise_std is not None:
            img += np.random.randn_like(img) * self.noise_std
        return img

    def __call__(self, img: Image):
        img_rs = img.resize((self.width, self.height), resample=self.interpolation)
        if not self.upscale:
            out = np.concatenate([img, np.zeros_like(img)], axis=self.axis)
            hf, ht = img.height, img.height + img_rs.height
            out[hf:ht, :img_rs.width] = self.add_noise(img_rs)
            return out
        else:
            return np.concatenate([img, img_rs.resize((img.size))], axis=self.axis)




def pil2np(img):
    '''Converts a PIL image to a numpy array.

    Parameters
    ----------
    img : PIL.Image
        Input image

    Returns
    -------
    np.ndarray
        Numpy array of image data.
    '''
    return np.array(img)


def add_tensor_noise(tens, noise_lvl=None) -> torch.Tensor:
    '''Adds noise to an image.

    Parameters
    ----------
    tens : torch.tensor
        Input image

    Returns
    -------
    torch.tensor
        Noisy version of input image.
    '''
    noise_lvl = _noise_STD if noise_lvl is None else noise_lvl
    eps = torch.randn_like(tens) * noise_lvl
    return tens + eps


def pil2blur(img, radius=1, axis=0):
    '''Converts a PIL image to a Gaussian blurred numpy array.

    Parameters
    ----------
    img : PIL.Image
        Input image

    radius : int, optional
        The blur radius of the Gaussian filter

    Returns
    -------
    np.ndarray
        Numpy array of image data.

    Notes
    -----
    By default the returned array is contactenated along the first axis, i.e. in the y axis in the image.
    In other words, if the input image is of dimension [m, n], the output dimensions are [2*m, n].
    The original image is thus contained in the [:m, :] block of the output, and the blurred image is contained in
    the [m:, :] block.
    '''
    img_blur = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.concatenate([img, img_blur], axis=axis)

def pil_grayscale(img) -> Image:
    '''Converts image to grayscale.

    Parameters
    ----------
    img : PIL.Image
        Numpy array of color image.

    Returns
    -------
    PIL.Image
        A grayscale version of an image.
    '''
    return img.convert('LA')

def pil_resize(img, scale=0.5, resample=Image.LANCZOS) -> Image:
    '''Resizes image. Uses

    Parameters
    ----------
    img : PIL.Image
        Numpy array of color image.

    Returns
    -------
    PIL.Image
        A resized version of an image.
    '''
    width, height = img.size
    width = int(width * scale)
    height = int(height * scale)
    return img.resize((width, height), resample=resample)


def get_random_vector(img, axis=1):
    '''Retrieves a random vector from a 2 dimensional image.

    This function randomly selects a row or column vector in an image and returns it.

    Parameters
    ----------
    img : PIL.Image
        The input image
    axis : int, optional
        The axis from whether to choose a vector.

    Returns
    -------
    A 1d vector randomly selected from an image.
    '''
    length = img.shape[axis]
    choice = np.random.choice(np.arange(length))
    out = np.take(img, choice, axis=axis)
    return out


def gaussian_blur_block_1d(x, k, sigma=1, axis=0):
    '''One dimensional Gaussian Blur transform.

    Parameters
    ----------
    x : np.ndarray
        Signal to convolve.

    k : int
        Kernel size.

    sigma : float
        Standard deviation.


    Returns
    -------
    A 1d convolved signal by a gaussian kernel
    '''
    flt = gaussian(k, sigma)
    flt /= np.sum(flt)
    if len(x.shape) > 1:
        out = np.zeros_like(x)
        for i in range(x.shape[1]):
            out[:,i] = np.convolve(x[:,i], flt, 'same')
        return np.expand_dims(np.concatenate([x, out], axis=axis), 1)

    out = np.convolve(x, flt, 'same')
    return np.expand_dims(np.concatenate([x, out], axis=axis), 0).astype(np.float32)