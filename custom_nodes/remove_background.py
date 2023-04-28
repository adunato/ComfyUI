# By Adunato 
#
# Copyright 2023 Adunato
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# tensor2pil and pil2tensor by WASasquatch

import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
from rembg import remove
import numpy as np
import math


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class RemoveBackground:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_remove_background"

    CATEGORY = "image/postprocessing"

    def image_remove_background(self, image):
        pil_image = tensor2pil(image)
        output_image = remove(pil_image)
        return (pil2tensor(output_image), )
    
class CropTransparent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_resolution_multi": ("INT", {"default": 8, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE_BOUNDS",)
    FUNCTION = "image_crop_transparent"

    CATEGORY = "image/postprocessing"

    def image_crop_transparent(self, image, image_resolution_multi):
        pil_image = tensor2pil(image)
        bbox = pil_image.getbbox()

        left, upper, right, lower = bbox
        width = right - left
        height = lower - upper

        # Round up dimensions to the nearest multiple of 64
        new_width = math.ceil(width / image_resolution_multi) * image_resolution_multi
        new_height = math.ceil(height / image_resolution_multi) * image_resolution_multi

        print(f"new_width: {new_width}, new_height: {new_height}")

        # Calculate the necessary padding
        pad_left = (new_width - width) // 2
        pad_upper = (new_height - height) // 2

        # Adjust the bounding box, extending the cropped area as needed
        new_left = max(left - pad_left, 0)
        new_upper = max(upper - pad_upper, 0)

        #if the boundaries of cropped area exceed the original image we reduce the selection to the next 64 multiple
        if new_left + new_width > pil_image.width:
            new_width = new_width - image_resolution_multi

        if new_upper + new_height > pil_image.height:
            new_height = new_height - image_resolution_multi

        print(f"reduced new_width: {new_width}, new_height: {new_height}")

        # Ensure dimensions are multiples of 32
        # new_width = min(new_width, pil_image.width - new_left)
        # new_height = min(new_height, pil_image.height - new_upper)

        new_right = new_left + new_width
        new_lower = new_upper + new_height

        # Crop the image using the adjusted bounding box
        output_image = pil_image.crop((new_left, new_upper, new_right, new_lower))
        image_bounds = [new_upper, new_lower, new_left, new_right]
        print(f"image_bounds: {image_bounds}")

        return (pil2tensor(output_image), image_bounds,)

class CalculateImageBounds:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image_bounds": ("IMAGE_BOUNDS",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE_BOUNDS",)
    FUNCTION = "calculate_image_bounds"

    CATEGORY = "image/postprocessing"

    def calculate_image_bounds(self, image, image_bounds):
        pil_image = tensor2pil(image)
        rmin, rmax, cmin, cmax = image_bounds
        output_image_bounds = [rmin, rmin+pil_image.height, cmin, cmin+pil_image.width]
        print(f"calculate_image_bounds: {output_image_bounds}")
        return (output_image_bounds,)


NODE_CLASS_MAPPINGS = {
    "RemoveBackground": RemoveBackground,
    "CropTransparent": CropTransparent,
    "CalculateImageBounds": CalculateImageBounds,
}