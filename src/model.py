from torchvision.transforms import functional as TF
from huggingface_hub import hf_hub_download
from PIL import Image

import torch.nn.functional as F
import torch.nn as nn
import torch


class NoImageError(Exception):
    """Exception raised when no input image is provided.

    This exception indicates that the colorization process was requested
    without a valid image object.
    """

    def __init__(self) -> None:
        """Initialize the exception with a descriptive error message."""
        super().__init__("No image found in request!")


class ConvBlock(nn.Module):
    """Convolutional block used in the U-Net architecture.

    The block consists of two convolutional layers, each followed by
    batch normalization and a ReLU activation function.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by each convolutional layer.

    Attributes
    ----------
    block : torch.nn.Sequential
        Sequential container holding the convolutional layers,
        batch normalization layers, and activation functions.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block to an input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C_in, H, W), where B is
            the batch size, C_in is the number of input channels,
            and H and W are the spatial dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C_out, H, W), where C_out
            equals out_channels. Spatial dimensions are preserved
            by the convolutional layers.
        """
        return self.block(x)


class UNetColorizer(nn.Module):
    """U-Net convolutional neural network for image colorization.

    The network transforms grayscale images into RGB images using an
    encoder-decoder architecture with skip connections. The encoder
    extracts hierarchical features, while the decoder reconstructs
    spatial details using transposed convolutions and concatenated
    encoder features.

    Parameters
    ----------
    in_channels : int, default=1
        Number of input image channels. The default value corresponds
        to grayscale images.
    out_channels : int, default=3
        Number of output image channels. The default value corresponds
        to RGB images.
    base_channels : int, default=32
        Number of feature channels in the first encoder block.
        The number of channels increases in subsequent encoder stages.

    Attributes
    ----------
    enc1 : ConvBlock
        First encoder block.
    enc2 : ConvBlock
        Second encoder block.
    enc3 : ConvBlock
        Third encoder block.
    pool : torch.nn.MaxPool2d
        Max-pooling layer used to reduce spatial dimensions.
    bottleneck : ConvBlock
        Bottleneck block connecting the encoder and decoder.
    up3 : torch.nn.ConvTranspose2d
        First decoder upsampling layer.
    dec3 : ConvBlock
        First decoder convolutional block.
    up2 : torch.nn.ConvTranspose2d
        Second decoder upsampling layer.
    dec2 : ConvBlock
        Second decoder convolutional block.
    up1 : torch.nn.ConvTranspose2d
        Final decoder upsampling layer.
    dec1 : ConvBlock
        Final decoder convolutional block.
    final : torch.nn.Conv2d
        Convolutional layer mapping decoder features to output channels.
    activation : torch.nn.Sigmoid
        Activation function constraining output values to the range
        [0, 1].
    """

    def __init__(
        self, in_channels: int = 1, out_channels: int = 3, base_channels: int = 32
    ) -> None:
        """Initialize the U-Net architecture.

        Parameters
        ----------
        in_channels : int, default=1
            Number of channels in the input tensor.
        out_channels : int, default=3
            Number of channels in the output tensor.
        base_channels : int, default=32
            Base number of feature channels used to determine the
            channel widths of the encoder, bottleneck, and decoder.
        """
        super().__init__()

        c = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, c)  # 96x96
        self.enc2 = ConvBlock(c, c * 2)  # 48x48
        self.enc3 = ConvBlock(c * 2, c * 4)  # 24x24

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = ConvBlock(c * 4, c * 8)  # 12x12

        # Decoder
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)  # 24x24

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)  # 48x48

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)  # 96x96

        # RGB output
        self.final = nn.Conv2d(c, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate an RGB prediction from a batch of grayscale images.

        The forward pass consists of three encoder stages, a bottleneck,
        and three decoder stages. Skip connections concatenate encoder
        features with the corresponding decoder features.

        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images with shape
            (B, in_channels, H, W). For the default architecture,
            the expected spatial dimensions are 96 x 96.

        Returns
        -------
        torch.Tensor
            Predicted RGB images with shape
            (B, out_channels, H, W). For the default architecture,
            the output shape is (B, 3, 96, 96) and values are
            constrained to the range [0, 1].
        """
        # Encoder
        e1 = self.enc1(x)  # (B, 32, 96, 96)
        e2 = self.enc2(self.pool(e1))  # (B, 64, 48, 48)
        e3 = self.enc3(self.pool(e2))  # (B, 128, 24, 24)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # (B, 256, 12, 12)

        # Decoder + skip connections
        d3 = self.up3(b)  # (B, 128, 24, 24)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)  # (B, 128, 24, 24)

        d2 = self.up2(d3)  # (B, 64, 48, 48)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)  # (B, 64, 48, 48)

        d1 = self.up1(d2)  # (B, 32, 96, 96)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)  # (B, 32, 96, 96)

        return self.activation(self.final(d1))  # (B, 3, 96, 96)


class Model:
    """High-level interface for loading and running the colorization model.

    This class manages device selection, checkpoint downloading and
    loading, neural network initialization, and image colorization.
    Large images are processed as overlapping tiles, and their predictions
    are blended to produce a continuous RGB image.

    Attributes
    ----------
    device : torch.device
        Device used for neural network inference, either CUDA or CPU.
    net : UNetColorizer
        Initialized U-Net model loaded with pretrained parameters and
        configured for evaluation.
    """

    def __init__(self) -> None:
        """Initialize the colorization model.

        Select the available computation device, download the pretrained
        checkpoint from the Hugging Face Hub, load the model parameters,
        and configure the network for inference.

        Notes
        -----
        The checkpoint is downloaded from the mzivro/my-models
        repository. The model is loaded in evaluation mode and moved
        to the selected device.

        Raises
        ------
        Exception
            Propagates exceptions raised during checkpoint downloading,
            loading, or model initialization.
        """
        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # download parameters
        checkpoint_path = hf_hub_download(
            repo_id="mzivro/my-models",
            filename="image_colorizer.pth",
            repo_type="model",
        )

        # load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )

        # load model and weights
        self.net = UNetColorizer()

        self.net.load_state_dict(
            checkpoint["model_state_dict"],
        )

        self.net.to(self.device)

        self.net.eval()

        print("Model ready")

    def __del__(self) -> None:
        """Release cached CUDA memory when the object is destroyed."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def colorize(
        self,
        image: Image.Image,
        tile_size: int = 96,
        overlap: int = 48,
        batch_size: int = 16,
    ) -> Image.Image:
        """Colorize an input image using tiled neural network inference.

        Convert the input image to grayscale, divide it into overlapping
        tiles, predict RGB values for each tile, and blend the predictions
        using spatial weights. Padding ensures that the tile grid covers
        the complete image. The final result is cropped to the original
        image dimensions.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image to colorize. The image is converted to RGB before
            processing and subsequently converted to grayscale.
        tile_size : int, default=96
            Height and width of each square input tile, in pixels.
            For the default network architecture, this should be 96.
        overlap : int, default=48
            Number of overlapping pixels between adjacent tiles.
            Must be smaller than tile_size for a positive stride.
            A positive overlap enables Hann-window blending.
        batch_size : int, default=16
            Maximum number of tiles processed in a single inference batch.
            Larger values may increase memory consumption.

        Returns
        -------
        PIL.Image.Image
            Colorized RGB image with the same width and height as the
            original input image.

        Raises
        ------
        NoImageError
            If image is falsy.
        ValueError
            If the image cannot be converted or processed, or if invalid
            tensor dimensions result from the supplied parameters.
        RuntimeError
            If PyTorch encounters an error during tensor operations
            or model inference.

        Notes
        -----
        Inference is performed with gradient computation disabled.
        Tile predictions are accumulated using a Hann window when
        overlap is positive; otherwise, uniform weights are used.
        Output pixel values are clamped to the range [0, 1].
        """
        if not image:
            raise NoImageError()

        image = image.convert("RGB")

        # load image and convert to grayscale tensor [1, H, W]
        original_width, original_height = image.size

        gray_image = image.convert("L")
        gray = TF.to_tensor(gray_image)

        _, height, width = gray.shape

        # Add padding so every pixel is covered by the tile grid
        # A margin moves original borders away from tile borders
        margin = overlap // 2
        stride = tile_size - overlap

        padded_width = max(
            tile_size,
            tile_size + ((width + margin - tile_size + stride - 1) // stride) * stride,
        )
        padded_height = max(
            tile_size,
            tile_size + ((height + margin - tile_size + stride - 1) // stride) * stride,
        )

        pad_right = padded_width - width - margin
        pad_bottom = padded_height - height - margin

        gray = F.pad(
            gray.unsqueeze(0),
            (margin, pad_right, margin, pad_bottom),
            mode="replicate",
        ).squeeze(0)

        # generate tile coordinates
        tiles = []

        for y in range(0, padded_height - tile_size + 1, stride):
            for x in range(0, padded_width - tile_size + 1, stride):
                tile = gray[
                    :,
                    y : y + tile_size,
                    x : x + tile_size,
                ]
                tiles.append((x, y, tile))

        # create blending weights
        # hann window reduces the contribution of tile edges
        if overlap > 0:
            window_1d = torch.hann_window(tile_size, periodic=False).clamp_min(0.05)

            weight = window_1d[:, None] * window_1d[None, :]
            weight = weight.unsqueeze(0)  # [1, H, W]
        else:
            weight = torch.ones(1, tile_size, tile_size)

        # accumulate predictions and weights
        prediction_sum = torch.zeros(3, padded_height, padded_width)
        weight_sum = torch.zeros(1, padded_height, padded_width)

        # predict tiles in batches
        with torch.inference_mode():
            for start in range(0, len(tiles), batch_size):
                batch_tiles = tiles[start : start + batch_size]

                inputs = torch.stack([tile for _, _, tile in batch_tiles]).to(
                    self.device
                )

                predictions = self.net(inputs).float().cpu().clamp(0, 1)

                for prediction, (x, y, _) in zip(predictions, batch_tiles):
                    prediction_sum[:, y : y + tile_size, x : x + tile_size] += (
                        prediction * weight
                    )

                    weight_sum[:, y : y + tile_size, x : x + tile_size] += weight

        # weighted blending
        result = prediction_sum / weight_sum.clamp_min(1e-8)

        # remove padding and restore original dimensions.
        result = result[
            :,
            margin : margin + original_height,
            margin : margin + original_width,
        ]

        result = result.clamp(0, 1)

        # convert [C, H, W] to PIL RGB
        result_image = TF.to_pil_image(result)

        return result_image
