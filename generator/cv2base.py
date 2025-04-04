dev = True
import time
import cv2
import numpy as np
import torch
from ast import literal_eval # required to enter types like Size as "[3, 5]"

if __name__ != "__main__":
	import comfy.utils
	import comfy.model_management as model_management
else:
	from PIL import Image
	from torchvision.transforms.functional import to_tensor

	class MockModelManagement:
		def __init__(self):
			self.device = torch.device('cpu')

		def get_torch_device(self):
			return self.device

	model_management = MockModelManagement()

class AnyType(str):
	def __ne__(self, __value: object) -> bool:
		return False

	def __eq__(self, __value: object) -> bool:
		return True

any = AnyType("*")

class Image2Nparray:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= "execute"
	CATEGORY	= "image/OpenCV"

	def execute(self, image):
		if len(image) > 1:
			raise Exception(f'Only images with batch_size==1 are supported! batch_size={len(image)}')
		ret = (image[0].cpu().numpy()[..., ::-1] * 255).astype(np.uint8) # reverse color channel from RGB to BGR
		return (ret,)

class Nparrays2Image:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"nparrays": ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("IMAGE",)
	RETURN_NAMES	= ("image",)
	FUNCTION	= "execute"
	CATEGORY	= "image/OpenCV"
	INPUT_IS_LIST	= True

	def execute(self, nparrays):
		image_hwcs = []
		for nparray in nparrays:
			if	len(nparray.shape)	== 2: nparray = cv2.cvtColor(nparray, cv2.COLOR_GRAY2RGB	)	# Grayscale image (H, W)
			elif	nparray.shape[2]	== 1: nparray = cv2.cvtColor(nparray, cv2.COLOR_GRAY2RGB	)	# Single-channel grayscale (H, W, 1)
			elif	nparray.shape[2]	== 3: nparray = cv2.cvtColor(nparray, cv2.COLOR_BGR2RGB	)	# BGR image (H, W, 3)

			image_hwc = torch.from_numpy(nparray.astype(np.float32) / 255.0).to(model_management.get_torch_device())
			image_hwcs.append(image_hwc)
		ret = torch.stack(image_hwcs)
		return (ret,)

def apply_function(func, func_args_in, image_idxs=[], literal_idxs=[]):
	#func_args = func_args_in.copy()
	func_args = func_args_in

	# evaluate literals
	for idx in literal_idxs:
		func_args[idx] = literal_eval(func_args[idx])

	output = func(*func_args)
	ret = output if isinstance(output, tuple) else (output,)
	return ret

NODE_DISPLAY_NAME_MAPPINGS	= {
	"Image2Nparray"	: "Image2Nparray",
	"Nparrays2Image"	: "Nparrays2Image",
}
NODE_CLASS_MAPPINGS	= {
	"Image2Nparray"	: Image2Nparray,
	"Nparrays2Image"	: Nparrays2Image,
}

if "dev" in globals() and dev:
	class cv2_SimpleTest:
		@classmethod
		def INPUT_TYPES(cls):
			return {
				'required': {
					"src"	: ("NPARRAY",),
				},
			}
		RETURN_TYPES	= ("INT",)
		RETURN_NAMES	= ("mean",)
		FUNCTION	= 'execute'
		CATEGORY	= 'image/OpenCV'

		def execute(self, src):
			ret = apply_function(cv2.mean, [src], [0], [])
			return ret

	NODE_DISPLAY_NAME_MAPPINGS	["cv2_SimpleTest"] = "cv2_SimpleTest"
	NODE_CLASS_MAPPINGS	["cv2_SimpleTest"] = cv2_SimpleTest

if __name__ == "__main__":
	img_pil_0 = Image.open("example.png")  # PIL Image (H, W, C)
	img_pil_1 = img_pil_0.rotate(90)
	img_pil_2 = img_pil_0.rotate(180)
	img_pil_3 = img_pil_0.rotate(270)

	# Convert PIL to tensor (shape: [C, H, W]), automatically scales to [0.0, 1.0]
	img_chw_0 = to_tensor(img_pil_0)
	img_chw_1 = to_tensor(img_pil_1)
	img_chw_2 = to_tensor(img_pil_2)
	img_chw_3 = to_tensor(img_pil_3)

	img_bchw = torch.stack([img_chw_0, img_chw_1, img_chw_2, img_chw_3], dim=0)	# (4, C, H, W)
	img_bwhc = img_bchw.permute(0, 2, 3, 1)	# (4, H, W, C)

	print(img_bwhc.shape)

	nparrays,	= Image2Nparray().execute(img_bwhc)
	kernel	= 5
	out_0	= apply_function(cv2.medianBlur, [nparrays, kernel], [0], [])
	out_1	= [cv2.cvtColor(nparray, cv2.COLOR_BGR2GRAY) for nparray in out_0[0]]
	image	= Nparrays2Image().execute(out_1)
