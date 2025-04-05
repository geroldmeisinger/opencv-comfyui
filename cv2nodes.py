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
class cv2_CamShift_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"probImage"	: ("NPARRAY",),
				"window"	: ("STRING",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING", "STRING",)
	RETURN_NAMES	= ("literal_0", "literal_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, probImage, window, criteria):
		ret = apply_function(cv2.CamShift, [probImage, window, criteria], [0], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["CamShift_0"] = "OpenCV CamShift_0"
NODE_CLASS_MAPPINGS	["CamShift_0"] = cv2_CamShift_0

class cv2_CamShift_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"probImage"	: ("NPARRAY",),
				"window"	: ("STRING",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING", "STRING",)
	RETURN_NAMES	= ("literal_0", "literal_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, probImage, window, criteria):
		ret = apply_function(cv2.CamShift, [probImage, window, criteria], [0], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["CamShift_1"] = "OpenCV CamShift_1"
NODE_CLASS_MAPPINGS	["CamShift_1"] = cv2_CamShift_1

class cv2_Canny_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"threshold1"	: ("FLOAT",),
				"threshold2"	: ("FLOAT",),
				"apertureSize"	: ("INT",),
				"L2gradient"	: ("BOOLEAN",),
			},
			'optional': {
				"edges"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, threshold1, threshold2, apertureSize, L2gradient, edges=None):
		ret = apply_function(cv2.Canny, [image, threshold1, threshold2, edges, apertureSize, L2gradient], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Canny_0"] = "OpenCV Canny_0"
NODE_CLASS_MAPPINGS	["Canny_0"] = cv2_Canny_0

class cv2_Canny_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"threshold1"	: ("FLOAT",),
				"threshold2"	: ("FLOAT",),
				"apertureSize"	: ("INT",),
				"L2gradient"	: ("BOOLEAN",),
			},
			'optional': {
				"edges"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, threshold1, threshold2, apertureSize, L2gradient, edges=None):
		ret = apply_function(cv2.Canny, [image, threshold1, threshold2, edges, apertureSize, L2gradient], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Canny_1"] = "OpenCV Canny_1"
NODE_CLASS_MAPPINGS	["Canny_1"] = cv2_Canny_1

class cv2_Canny_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"dx"	: ("NPARRAY",),
				"dy"	: ("NPARRAY",),
				"threshold1"	: ("FLOAT",),
				"threshold2"	: ("FLOAT",),
				"L2gradient"	: ("BOOLEAN",),
			},
			'optional': {
				"edges"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, dx, dy, threshold1, threshold2, L2gradient, edges=None):
		ret = apply_function(cv2.Canny, [dx, dy, threshold1, threshold2, edges, L2gradient], [0, 1, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Canny_2"] = "OpenCV Canny_2"
NODE_CLASS_MAPPINGS	["Canny_2"] = cv2_Canny_2

class cv2_Canny_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"dx"	: ("NPARRAY",),
				"dy"	: ("NPARRAY",),
				"threshold1"	: ("FLOAT",),
				"threshold2"	: ("FLOAT",),
				"L2gradient"	: ("BOOLEAN",),
			},
			'optional': {
				"edges"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, dx, dy, threshold1, threshold2, L2gradient, edges=None):
		ret = apply_function(cv2.Canny, [dx, dy, threshold1, threshold2, edges, L2gradient], [0, 1, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Canny_3"] = "OpenCV Canny_3"
NODE_CLASS_MAPPINGS	["Canny_3"] = cv2_Canny_3

class cv2_EMD_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"signature1"	: ("NPARRAY",),
				"signature2"	: ("NPARRAY",),
				"distType"	: ("INT",),
			},
			'optional': {
				"cost"	: ("NPARRAY",),
				"cost"	: ("NPARRAY",),
				"flow"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float_0", "float_1", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, signature1, signature2, distType, cost=None, lowerBound=None, flow=None):
		ret = apply_function(cv2.EMD, [signature1, signature2, distType, cost, lowerBound, flow], [0, 1, 3, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["EMD_0"] = "OpenCV EMD_0"
NODE_CLASS_MAPPINGS	["EMD_0"] = cv2_EMD_0

class cv2_EMD_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"signature1"	: ("NPARRAY",),
				"signature2"	: ("NPARRAY",),
				"distType"	: ("INT",),
			},
			'optional': {
				"cost"	: ("NPARRAY",),
				"cost"	: ("NPARRAY",),
				"flow"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float_0", "float_1", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, signature1, signature2, distType, cost=None, lowerBound=None, flow=None):
		ret = apply_function(cv2.EMD, [signature1, signature2, distType, cost, lowerBound, flow], [0, 1, 3, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["EMD_1"] = "OpenCV EMD_1"
NODE_CLASS_MAPPINGS	["EMD_1"] = cv2_EMD_1

class cv2_GaussianBlur_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("STRING",),
				"sigmaX"	: ("FLOAT",),
				"sigmaY"	: ("FLOAT",),
				"borderType"	: ("INT",),
				"hint"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, sigmaX, sigmaY, borderType, hint, dst=None):
		ret = apply_function(cv2.GaussianBlur, [src, ksize, sigmaX, dst, sigmaY, borderType, hint], [0, 3], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["GaussianBlur_0"] = "OpenCV GaussianBlur_0"
NODE_CLASS_MAPPINGS	["GaussianBlur_0"] = cv2_GaussianBlur_0

class cv2_GaussianBlur_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("STRING",),
				"sigmaX"	: ("FLOAT",),
				"sigmaY"	: ("FLOAT",),
				"borderType"	: ("INT",),
				"hint"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, sigmaX, sigmaY, borderType, hint, dst=None):
		ret = apply_function(cv2.GaussianBlur, [src, ksize, sigmaX, dst, sigmaY, borderType, hint], [0, 3], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["GaussianBlur_1"] = "OpenCV GaussianBlur_1"
NODE_CLASS_MAPPINGS	["GaussianBlur_1"] = cv2_GaussianBlur_1

class cv2_HoughCircles_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"method"	: ("INT",),
				"dp"	: ("FLOAT",),
				"minDist"	: ("FLOAT",),
				"param1"	: ("FLOAT",),
				"param2"	: ("FLOAT",),
				"minRadius"	: ("INT",),
				"maxRadius"	: ("INT",),
			},
			'optional': {
				"circles"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, method, dp, minDist, param1, param2, minRadius, maxRadius, circles=None):
		ret = apply_function(cv2.HoughCircles, [image, method, dp, minDist, circles, param1, param2, minRadius, maxRadius], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HoughCircles_0"] = "OpenCV HoughCircles_0"
NODE_CLASS_MAPPINGS	["HoughCircles_0"] = cv2_HoughCircles_0

class cv2_HoughCircles_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"method"	: ("INT",),
				"dp"	: ("FLOAT",),
				"minDist"	: ("FLOAT",),
				"param1"	: ("FLOAT",),
				"param2"	: ("FLOAT",),
				"minRadius"	: ("INT",),
				"maxRadius"	: ("INT",),
			},
			'optional': {
				"circles"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, method, dp, minDist, param1, param2, minRadius, maxRadius, circles=None):
		ret = apply_function(cv2.HoughCircles, [image, method, dp, minDist, circles, param1, param2, minRadius, maxRadius], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HoughCircles_1"] = "OpenCV HoughCircles_1"
NODE_CLASS_MAPPINGS	["HoughCircles_1"] = cv2_HoughCircles_1

class cv2_HoughLines_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"rho"	: ("FLOAT",),
				"theta"	: ("FLOAT",),
				"threshold"	: ("INT",),
				"srn"	: ("FLOAT",),
				"stn"	: ("FLOAT",),
				"min_theta"	: ("FLOAT",),
				"max_theta"	: ("FLOAT",),
				"use_edgeval"	: ("BOOLEAN",),
			},
			'optional': {
				"lines"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, rho, theta, threshold, srn, stn, min_theta, max_theta, use_edgeval, lines=None):
		ret = apply_function(cv2.HoughLines, [image, rho, theta, threshold, lines, srn, stn, min_theta, max_theta, use_edgeval], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HoughLines_0"] = "OpenCV HoughLines_0"
NODE_CLASS_MAPPINGS	["HoughLines_0"] = cv2_HoughLines_0

class cv2_HoughLines_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"rho"	: ("FLOAT",),
				"theta"	: ("FLOAT",),
				"threshold"	: ("INT",),
				"srn"	: ("FLOAT",),
				"stn"	: ("FLOAT",),
				"min_theta"	: ("FLOAT",),
				"max_theta"	: ("FLOAT",),
				"use_edgeval"	: ("BOOLEAN",),
			},
			'optional': {
				"lines"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, rho, theta, threshold, srn, stn, min_theta, max_theta, use_edgeval, lines=None):
		ret = apply_function(cv2.HoughLines, [image, rho, theta, threshold, lines, srn, stn, min_theta, max_theta, use_edgeval], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HoughLines_1"] = "OpenCV HoughLines_1"
NODE_CLASS_MAPPINGS	["HoughLines_1"] = cv2_HoughLines_1

class cv2_HoughLinesP_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"rho"	: ("FLOAT",),
				"theta"	: ("FLOAT",),
				"threshold"	: ("INT",),
				"minLineLength"	: ("FLOAT",),
				"maxLineGap"	: ("FLOAT",),
			},
			'optional': {
				"lines"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, rho, theta, threshold, minLineLength, maxLineGap, lines=None):
		ret = apply_function(cv2.HoughLinesP, [image, rho, theta, threshold, lines, minLineLength, maxLineGap], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HoughLinesP_0"] = "OpenCV HoughLinesP_0"
NODE_CLASS_MAPPINGS	["HoughLinesP_0"] = cv2_HoughLinesP_0

class cv2_HoughLinesP_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"rho"	: ("FLOAT",),
				"theta"	: ("FLOAT",),
				"threshold"	: ("INT",),
				"minLineLength"	: ("FLOAT",),
				"maxLineGap"	: ("FLOAT",),
			},
			'optional': {
				"lines"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, rho, theta, threshold, minLineLength, maxLineGap, lines=None):
		ret = apply_function(cv2.HoughLinesP, [image, rho, theta, threshold, lines, minLineLength, maxLineGap], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HoughLinesP_1"] = "OpenCV HoughLinesP_1"
NODE_CLASS_MAPPINGS	["HoughLinesP_1"] = cv2_HoughLinesP_1

class cv2_HoughLinesPointSet_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"point"	: ("NPARRAY",),
				"lines_max"	: ("INT",),
				"threshold"	: ("INT",),
				"min_rho"	: ("FLOAT",),
				"max_rho"	: ("FLOAT",),
				"rho_step"	: ("FLOAT",),
				"min_theta"	: ("FLOAT",),
				"max_theta"	: ("FLOAT",),
				"theta_step"	: ("FLOAT",),
			},
			'optional': {
				"lines"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, point, lines_max, threshold, min_rho, max_rho, rho_step, min_theta, max_theta, theta_step, lines=None):
		ret = apply_function(cv2.HoughLinesPointSet, [point, lines_max, threshold, min_rho, max_rho, rho_step, min_theta, max_theta, theta_step, lines], [0, 9], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HoughLinesPointSet_0"] = "OpenCV HoughLinesPointSet_0"
NODE_CLASS_MAPPINGS	["HoughLinesPointSet_0"] = cv2_HoughLinesPointSet_0

class cv2_HoughLinesPointSet_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"point"	: ("NPARRAY",),
				"lines_max"	: ("INT",),
				"threshold"	: ("INT",),
				"min_rho"	: ("FLOAT",),
				"max_rho"	: ("FLOAT",),
				"rho_step"	: ("FLOAT",),
				"min_theta"	: ("FLOAT",),
				"max_theta"	: ("FLOAT",),
				"theta_step"	: ("FLOAT",),
			},
			'optional': {
				"lines"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, point, lines_max, threshold, min_rho, max_rho, rho_step, min_theta, max_theta, theta_step, lines=None):
		ret = apply_function(cv2.HoughLinesPointSet, [point, lines_max, threshold, min_rho, max_rho, rho_step, min_theta, max_theta, theta_step, lines], [0, 9], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HoughLinesPointSet_1"] = "OpenCV HoughLinesPointSet_1"
NODE_CLASS_MAPPINGS	["HoughLinesPointSet_1"] = cv2_HoughLinesPointSet_1

class cv2_HoughLinesWithAccumulator_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"rho"	: ("FLOAT",),
				"theta"	: ("FLOAT",),
				"threshold"	: ("INT",),
				"srn"	: ("FLOAT",),
				"stn"	: ("FLOAT",),
				"min_theta"	: ("FLOAT",),
				"max_theta"	: ("FLOAT",),
			},
			'optional': {
				"lines"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, rho, theta, threshold, srn, stn, min_theta, max_theta, lines=None):
		ret = apply_function(cv2.HoughLinesWithAccumulator, [image, rho, theta, threshold, lines, srn, stn, min_theta, max_theta], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HoughLinesWithAccumulator_0"] = "OpenCV HoughLinesWithAccumulator_0"
NODE_CLASS_MAPPINGS	["HoughLinesWithAccumulator_0"] = cv2_HoughLinesWithAccumulator_0

class cv2_HoughLinesWithAccumulator_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"rho"	: ("FLOAT",),
				"theta"	: ("FLOAT",),
				"threshold"	: ("INT",),
				"srn"	: ("FLOAT",),
				"stn"	: ("FLOAT",),
				"min_theta"	: ("FLOAT",),
				"max_theta"	: ("FLOAT",),
			},
			'optional': {
				"lines"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, rho, theta, threshold, srn, stn, min_theta, max_theta, lines=None):
		ret = apply_function(cv2.HoughLinesWithAccumulator, [image, rho, theta, threshold, lines, srn, stn, min_theta, max_theta], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HoughLinesWithAccumulator_1"] = "OpenCV HoughLinesWithAccumulator_1"
NODE_CLASS_MAPPINGS	["HoughLinesWithAccumulator_1"] = cv2_HoughLinesWithAccumulator_1

class cv2_HuMoments_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"m"	: ("STRING",),
			},
			'optional': {
				"hu"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, m, hu=None):
		ret = apply_function(cv2.HuMoments, [m, hu], [1], [0])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HuMoments_0"] = "OpenCV HuMoments_0"
NODE_CLASS_MAPPINGS	["HuMoments_0"] = cv2_HuMoments_0

class cv2_HuMoments_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"m"	: ("STRING",),
			},
			'optional': {
				"hu"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, m, hu=None):
		ret = apply_function(cv2.HuMoments, [m, hu], [1], [0])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["HuMoments_1"] = "OpenCV HuMoments_1"
NODE_CLASS_MAPPINGS	["HuMoments_1"] = cv2_HuMoments_1

class cv2_LUT_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"lut"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, lut, dst=None):
		ret = apply_function(cv2.LUT, [src, lut, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["LUT_0"] = "OpenCV LUT_0"
NODE_CLASS_MAPPINGS	["LUT_0"] = cv2_LUT_0

class cv2_LUT_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"lut"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, lut, dst=None):
		ret = apply_function(cv2.LUT, [src, lut, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["LUT_1"] = "OpenCV LUT_1"
NODE_CLASS_MAPPINGS	["LUT_1"] = cv2_LUT_1

class cv2_Laplacian_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"ksize"	: ("INT",),
				"scale"	: ("FLOAT",),
				"delta"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, ksize, scale, delta, borderType, dst=None):
		ret = apply_function(cv2.Laplacian, [src, ddepth, dst, ksize, scale, delta, borderType], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Laplacian_0"] = "OpenCV Laplacian_0"
NODE_CLASS_MAPPINGS	["Laplacian_0"] = cv2_Laplacian_0

class cv2_Laplacian_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"ksize"	: ("INT",),
				"scale"	: ("FLOAT",),
				"delta"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, ksize, scale, delta, borderType, dst=None):
		ret = apply_function(cv2.Laplacian, [src, ddepth, dst, ksize, scale, delta, borderType], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Laplacian_1"] = "OpenCV Laplacian_1"
NODE_CLASS_MAPPINGS	["Laplacian_1"] = cv2_Laplacian_1

class cv2_Mahalanobis_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"v1"	: ("NPARRAY",),
				"v2"	: ("NPARRAY",),
				"icovar"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, v1, v2, icovar):
		ret = apply_function(cv2.Mahalanobis, [v1, v2, icovar], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Mahalanobis_0"] = "OpenCV Mahalanobis_0"
NODE_CLASS_MAPPINGS	["Mahalanobis_0"] = cv2_Mahalanobis_0

class cv2_Mahalanobis_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"v1"	: ("NPARRAY",),
				"v2"	: ("NPARRAY",),
				"icovar"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, v1, v2, icovar):
		ret = apply_function(cv2.Mahalanobis, [v1, v2, icovar], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Mahalanobis_1"] = "OpenCV Mahalanobis_1"
NODE_CLASS_MAPPINGS	["Mahalanobis_1"] = cv2_Mahalanobis_1

class cv2_PCABackProject_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"eigenvectors"	: ("NPARRAY",),
			},
			'optional': {
				"result"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, eigenvectors, result=None):
		ret = apply_function(cv2.PCABackProject, [data, mean, eigenvectors, result], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCABackProject_0"] = "OpenCV PCABackProject_0"
NODE_CLASS_MAPPINGS	["PCABackProject_0"] = cv2_PCABackProject_0

class cv2_PCABackProject_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"eigenvectors"	: ("NPARRAY",),
			},
			'optional': {
				"result"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, eigenvectors, result=None):
		ret = apply_function(cv2.PCABackProject, [data, mean, eigenvectors, result], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCABackProject_1"] = "OpenCV PCABackProject_1"
NODE_CLASS_MAPPINGS	["PCABackProject_1"] = cv2_PCABackProject_1

class cv2_PCACompute_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"maxComponents"	: ("INT",),
			},
			'optional': {
				"eigenvectors"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, maxComponents, eigenvectors=None):
		ret = apply_function(cv2.PCACompute, [data, mean, eigenvectors, maxComponents], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCACompute_0"] = "OpenCV PCACompute_0"
NODE_CLASS_MAPPINGS	["PCACompute_0"] = cv2_PCACompute_0

class cv2_PCACompute_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"maxComponents"	: ("INT",),
			},
			'optional': {
				"eigenvectors"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, maxComponents, eigenvectors=None):
		ret = apply_function(cv2.PCACompute, [data, mean, eigenvectors, maxComponents], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCACompute_1"] = "OpenCV PCACompute_1"
NODE_CLASS_MAPPINGS	["PCACompute_1"] = cv2_PCACompute_1

class cv2_PCACompute_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"retainedVariance"	: ("FLOAT",),
			},
			'optional': {
				"eigenvectors"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, retainedVariance, eigenvectors=None):
		ret = apply_function(cv2.PCACompute, [data, mean, retainedVariance, eigenvectors], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCACompute_2"] = "OpenCV PCACompute_2"
NODE_CLASS_MAPPINGS	["PCACompute_2"] = cv2_PCACompute_2

class cv2_PCACompute_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"retainedVariance"	: ("FLOAT",),
			},
			'optional': {
				"eigenvectors"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, retainedVariance, eigenvectors=None):
		ret = apply_function(cv2.PCACompute, [data, mean, retainedVariance, eigenvectors], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCACompute_3"] = "OpenCV PCACompute_3"
NODE_CLASS_MAPPINGS	["PCACompute_3"] = cv2_PCACompute_3

class cv2_PCACompute2_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"maxComponents"	: ("INT",),
			},
			'optional': {
				"eigenvectors"	: ("NPARRAY",),
				"eigenvalues"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, maxComponents, eigenvectors=None, eigenvalues=None):
		ret = apply_function(cv2.PCACompute2, [data, mean, eigenvectors, eigenvalues, maxComponents], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCACompute2_0"] = "OpenCV PCACompute2_0"
NODE_CLASS_MAPPINGS	["PCACompute2_0"] = cv2_PCACompute2_0

class cv2_PCACompute2_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"maxComponents"	: ("INT",),
			},
			'optional': {
				"eigenvectors"	: ("NPARRAY",),
				"eigenvalues"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, maxComponents, eigenvectors=None, eigenvalues=None):
		ret = apply_function(cv2.PCACompute2, [data, mean, eigenvectors, eigenvalues, maxComponents], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCACompute2_1"] = "OpenCV PCACompute2_1"
NODE_CLASS_MAPPINGS	["PCACompute2_1"] = cv2_PCACompute2_1

class cv2_PCACompute2_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"retainedVariance"	: ("FLOAT",),
			},
			'optional': {
				"eigenvectors"	: ("NPARRAY",),
				"eigenvalues"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, retainedVariance, eigenvectors=None, eigenvalues=None):
		ret = apply_function(cv2.PCACompute2, [data, mean, retainedVariance, eigenvectors, eigenvalues], [0, 1, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCACompute2_2"] = "OpenCV PCACompute2_2"
NODE_CLASS_MAPPINGS	["PCACompute2_2"] = cv2_PCACompute2_2

class cv2_PCACompute2_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"retainedVariance"	: ("FLOAT",),
			},
			'optional': {
				"eigenvectors"	: ("NPARRAY",),
				"eigenvalues"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, retainedVariance, eigenvectors=None, eigenvalues=None):
		ret = apply_function(cv2.PCACompute2, [data, mean, retainedVariance, eigenvectors, eigenvalues], [0, 1, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCACompute2_3"] = "OpenCV PCACompute2_3"
NODE_CLASS_MAPPINGS	["PCACompute2_3"] = cv2_PCACompute2_3

class cv2_PCAProject_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"eigenvectors"	: ("NPARRAY",),
			},
			'optional': {
				"result"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, eigenvectors, result=None):
		ret = apply_function(cv2.PCAProject, [data, mean, eigenvectors, result], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCAProject_0"] = "OpenCV PCAProject_0"
NODE_CLASS_MAPPINGS	["PCAProject_0"] = cv2_PCAProject_0

class cv2_PCAProject_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"eigenvectors"	: ("NPARRAY",),
			},
			'optional': {
				"result"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, mean, eigenvectors, result=None):
		ret = apply_function(cv2.PCAProject, [data, mean, eigenvectors, result], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PCAProject_1"] = "OpenCV PCAProject_1"
NODE_CLASS_MAPPINGS	["PCAProject_1"] = cv2_PCAProject_1

class cv2_PSNR_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"R"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, R):
		ret = apply_function(cv2.PSNR, [src1, src2, R], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PSNR_0"] = "OpenCV PSNR_0"
NODE_CLASS_MAPPINGS	["PSNR_0"] = cv2_PSNR_0

class cv2_PSNR_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"R"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, R):
		ret = apply_function(cv2.PSNR, [src1, src2, R], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["PSNR_1"] = "OpenCV PSNR_1"
NODE_CLASS_MAPPINGS	["PSNR_1"] = cv2_PSNR_1

class cv2_RQDecomp3x3_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"mtxR"	: ("NPARRAY",),
				"mtxQ"	: ("NPARRAY",),
				"Qx"	: ("NPARRAY",),
				"Qy"	: ("NPARRAY",),
				"Qz"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("None", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("unknown", "nparray_1", "nparray_2", "nparray_3", "nparray_4", "nparray_5",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mtxR=None, mtxQ=None, Qx=None, Qy=None, Qz=None):
		ret = apply_function(cv2.RQDecomp3x3, [src, mtxR, mtxQ, Qx, Qy, Qz], [0, 1, 2, 3, 4, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["RQDecomp3x3_0"] = "OpenCV RQDecomp3x3_0"
NODE_CLASS_MAPPINGS	["RQDecomp3x3_0"] = cv2_RQDecomp3x3_0

class cv2_RQDecomp3x3_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"mtxR"	: ("NPARRAY",),
				"mtxQ"	: ("NPARRAY",),
				"Qx"	: ("NPARRAY",),
				"Qy"	: ("NPARRAY",),
				"Qz"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("None", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("unknown", "nparray_1", "nparray_2", "nparray_3", "nparray_4", "nparray_5",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mtxR=None, mtxQ=None, Qx=None, Qy=None, Qz=None):
		ret = apply_function(cv2.RQDecomp3x3, [src, mtxR, mtxQ, Qx, Qy, Qz], [0, 1, 2, 3, 4, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["RQDecomp3x3_1"] = "OpenCV RQDecomp3x3_1"
NODE_CLASS_MAPPINGS	["RQDecomp3x3_1"] = cv2_RQDecomp3x3_1

class cv2_Rodrigues_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"jacobian"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None, jacobian=None):
		ret = apply_function(cv2.Rodrigues, [src, dst, jacobian], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Rodrigues_0"] = "OpenCV Rodrigues_0"
NODE_CLASS_MAPPINGS	["Rodrigues_0"] = cv2_Rodrigues_0

class cv2_Rodrigues_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"jacobian"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None, jacobian=None):
		ret = apply_function(cv2.Rodrigues, [src, dst, jacobian], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Rodrigues_1"] = "OpenCV Rodrigues_1"
NODE_CLASS_MAPPINGS	["Rodrigues_1"] = cv2_Rodrigues_1

class cv2_SVBackSubst_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"w"	: ("NPARRAY",),
				"u"	: ("NPARRAY",),
				"vt"	: ("NPARRAY",),
				"rhs"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, w, u, vt, rhs, dst=None):
		ret = apply_function(cv2.SVBackSubst, [w, u, vt, rhs, dst], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["SVBackSubst_0"] = "OpenCV SVBackSubst_0"
NODE_CLASS_MAPPINGS	["SVBackSubst_0"] = cv2_SVBackSubst_0

class cv2_SVBackSubst_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"w"	: ("NPARRAY",),
				"u"	: ("NPARRAY",),
				"vt"	: ("NPARRAY",),
				"rhs"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, w, u, vt, rhs, dst=None):
		ret = apply_function(cv2.SVBackSubst, [w, u, vt, rhs, dst], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["SVBackSubst_1"] = "OpenCV SVBackSubst_1"
NODE_CLASS_MAPPINGS	["SVBackSubst_1"] = cv2_SVBackSubst_1

class cv2_SVDecomp_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"w"	: ("NPARRAY",),
				"u"	: ("NPARRAY",),
				"vt"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, w=None, u=None, vt=None):
		ret = apply_function(cv2.SVDecomp, [src, w, u, vt, flags], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["SVDecomp_0"] = "OpenCV SVDecomp_0"
NODE_CLASS_MAPPINGS	["SVDecomp_0"] = cv2_SVDecomp_0

class cv2_SVDecomp_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"w"	: ("NPARRAY",),
				"u"	: ("NPARRAY",),
				"vt"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, w=None, u=None, vt=None):
		ret = apply_function(cv2.SVDecomp, [src, w, u, vt, flags], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["SVDecomp_1"] = "OpenCV SVDecomp_1"
NODE_CLASS_MAPPINGS	["SVDecomp_1"] = cv2_SVDecomp_1

class cv2_Scharr_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"dx"	: ("INT",),
				"dy"	: ("INT",),
				"scale"	: ("FLOAT",),
				"delta"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, dx, dy, scale, delta, borderType, dst=None):
		ret = apply_function(cv2.Scharr, [src, ddepth, dx, dy, dst, scale, delta, borderType], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Scharr_0"] = "OpenCV Scharr_0"
NODE_CLASS_MAPPINGS	["Scharr_0"] = cv2_Scharr_0

class cv2_Scharr_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"dx"	: ("INT",),
				"dy"	: ("INT",),
				"scale"	: ("FLOAT",),
				"delta"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, dx, dy, scale, delta, borderType, dst=None):
		ret = apply_function(cv2.Scharr, [src, ddepth, dx, dy, dst, scale, delta, borderType], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Scharr_1"] = "OpenCV Scharr_1"
NODE_CLASS_MAPPINGS	["Scharr_1"] = cv2_Scharr_1

class cv2_Sobel_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"dx"	: ("INT",),
				"dy"	: ("INT",),
				"ksize"	: ("INT",),
				"scale"	: ("FLOAT",),
				"delta"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, dx, dy, ksize, scale, delta, borderType, dst=None):
		ret = apply_function(cv2.Sobel, [src, ddepth, dx, dy, dst, ksize, scale, delta, borderType], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Sobel_0"] = "OpenCV Sobel_0"
NODE_CLASS_MAPPINGS	["Sobel_0"] = cv2_Sobel_0

class cv2_Sobel_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"dx"	: ("INT",),
				"dy"	: ("INT",),
				"ksize"	: ("INT",),
				"scale"	: ("FLOAT",),
				"delta"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, dx, dy, ksize, scale, delta, borderType, dst=None):
		ret = apply_function(cv2.Sobel, [src, ddepth, dx, dy, dst, ksize, scale, delta, borderType], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["Sobel_1"] = "OpenCV Sobel_1"
NODE_CLASS_MAPPINGS	["Sobel_1"] = cv2_Sobel_1

class cv2_absdiff_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None):
		ret = apply_function(cv2.absdiff, [src1, src2, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["absdiff_0"] = "OpenCV absdiff_0"
NODE_CLASS_MAPPINGS	["absdiff_0"] = cv2_absdiff_0

class cv2_absdiff_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None):
		ret = apply_function(cv2.absdiff, [src1, src2, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["absdiff_1"] = "OpenCV absdiff_1"
NODE_CLASS_MAPPINGS	["absdiff_1"] = cv2_absdiff_1

class cv2_accumulate_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, mask=None):
		ret = apply_function(cv2.accumulate, [src, dst, mask], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["accumulate_0"] = "OpenCV accumulate_0"
NODE_CLASS_MAPPINGS	["accumulate_0"] = cv2_accumulate_0

class cv2_accumulate_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, mask=None):
		ret = apply_function(cv2.accumulate, [src, dst, mask], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["accumulate_1"] = "OpenCV accumulate_1"
NODE_CLASS_MAPPINGS	["accumulate_1"] = cv2_accumulate_1

class cv2_accumulateProduct_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst, mask=None):
		ret = apply_function(cv2.accumulateProduct, [src1, src2, dst, mask], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["accumulateProduct_0"] = "OpenCV accumulateProduct_0"
NODE_CLASS_MAPPINGS	["accumulateProduct_0"] = cv2_accumulateProduct_0

class cv2_accumulateProduct_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst, mask=None):
		ret = apply_function(cv2.accumulateProduct, [src1, src2, dst, mask], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["accumulateProduct_1"] = "OpenCV accumulateProduct_1"
NODE_CLASS_MAPPINGS	["accumulateProduct_1"] = cv2_accumulateProduct_1

class cv2_accumulateSquare_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, mask=None):
		ret = apply_function(cv2.accumulateSquare, [src, dst, mask], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["accumulateSquare_0"] = "OpenCV accumulateSquare_0"
NODE_CLASS_MAPPINGS	["accumulateSquare_0"] = cv2_accumulateSquare_0

class cv2_accumulateSquare_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, mask=None):
		ret = apply_function(cv2.accumulateSquare, [src, dst, mask], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["accumulateSquare_1"] = "OpenCV accumulateSquare_1"
NODE_CLASS_MAPPINGS	["accumulateSquare_1"] = cv2_accumulateSquare_1

class cv2_accumulateWeighted_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, alpha, mask=None):
		ret = apply_function(cv2.accumulateWeighted, [src, dst, alpha, mask], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["accumulateWeighted_0"] = "OpenCV accumulateWeighted_0"
NODE_CLASS_MAPPINGS	["accumulateWeighted_0"] = cv2_accumulateWeighted_0

class cv2_accumulateWeighted_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, alpha, mask=None):
		ret = apply_function(cv2.accumulateWeighted, [src, dst, alpha, mask], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["accumulateWeighted_1"] = "OpenCV accumulateWeighted_1"
NODE_CLASS_MAPPINGS	["accumulateWeighted_1"] = cv2_accumulateWeighted_1

class cv2_adaptiveThreshold_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"maxValue"	: ("FLOAT",),
				"adaptiveMethod"	: ("INT",),
				"thresholdType"	: ("INT",),
				"blockSize"	: ("INT",),
				"C"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None):
		ret = apply_function(cv2.adaptiveThreshold, [src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst], [0, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["adaptiveThreshold_0"] = "OpenCV adaptiveThreshold_0"
NODE_CLASS_MAPPINGS	["adaptiveThreshold_0"] = cv2_adaptiveThreshold_0

class cv2_adaptiveThreshold_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"maxValue"	: ("FLOAT",),
				"adaptiveMethod"	: ("INT",),
				"thresholdType"	: ("INT",),
				"blockSize"	: ("INT",),
				"C"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None):
		ret = apply_function(cv2.adaptiveThreshold, [src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst], [0, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["adaptiveThreshold_1"] = "OpenCV adaptiveThreshold_1"
NODE_CLASS_MAPPINGS	["adaptiveThreshold_1"] = cv2_adaptiveThreshold_1

class cv2_add_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dtype, dst=None, mask=None):
		ret = apply_function(cv2.add, [src1, src2, dst, mask, dtype], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["add_0"] = "OpenCV add_0"
NODE_CLASS_MAPPINGS	["add_0"] = cv2_add_0

class cv2_add_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dtype, dst=None, mask=None):
		ret = apply_function(cv2.add, [src1, src2, dst, mask, dtype], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["add_1"] = "OpenCV add_1"
NODE_CLASS_MAPPINGS	["add_1"] = cv2_add_1

class cv2_addText_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"text"	: ("STRING",),
				"org"	: ("STRING",),
				"nameFont"	: ("STRING",),
				"pointSize"	: ("INT",),
				"color"	: ("STRING",),
				"weight"	: ("INT",),
				"style"	: ("INT",),
				"spacing"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, text, org, nameFont, pointSize, color, weight, style, spacing):
		ret = apply_function(cv2.addText, [img, text, org, nameFont, pointSize, color, weight, style, spacing], [0], [2, 5])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["addText_0"] = "OpenCV addText_0"
NODE_CLASS_MAPPINGS	["addText_0"] = cv2_addText_0

class cv2_addWeighted_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"src2"	: ("NPARRAY",),
				"beta"	: ("FLOAT",),
				"gamma"	: ("FLOAT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, alpha, src2, beta, gamma, dtype, dst=None):
		ret = apply_function(cv2.addWeighted, [src1, alpha, src2, beta, gamma, dst, dtype], [0, 2, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["addWeighted_0"] = "OpenCV addWeighted_0"
NODE_CLASS_MAPPINGS	["addWeighted_0"] = cv2_addWeighted_0

class cv2_addWeighted_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"src2"	: ("NPARRAY",),
				"beta"	: ("FLOAT",),
				"gamma"	: ("FLOAT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, alpha, src2, beta, gamma, dtype, dst=None):
		ret = apply_function(cv2.addWeighted, [src1, alpha, src2, beta, gamma, dst, dtype], [0, 2, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["addWeighted_1"] = "OpenCV addWeighted_1"
NODE_CLASS_MAPPINGS	["addWeighted_1"] = cv2_addWeighted_1

class cv2_applyColorMap_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"colormap"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, colormap, dst=None):
		ret = apply_function(cv2.applyColorMap, [src, colormap, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["applyColorMap_0"] = "OpenCV applyColorMap_0"
NODE_CLASS_MAPPINGS	["applyColorMap_0"] = cv2_applyColorMap_0

class cv2_applyColorMap_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"colormap"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, colormap, dst=None):
		ret = apply_function(cv2.applyColorMap, [src, colormap, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["applyColorMap_1"] = "OpenCV applyColorMap_1"
NODE_CLASS_MAPPINGS	["applyColorMap_1"] = cv2_applyColorMap_1

class cv2_applyColorMap_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"userColor"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, userColor, dst=None):
		ret = apply_function(cv2.applyColorMap, [src, userColor, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["applyColorMap_2"] = "OpenCV applyColorMap_2"
NODE_CLASS_MAPPINGS	["applyColorMap_2"] = cv2_applyColorMap_2

class cv2_applyColorMap_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"userColor"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, userColor, dst=None):
		ret = apply_function(cv2.applyColorMap, [src, userColor, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["applyColorMap_3"] = "OpenCV applyColorMap_3"
NODE_CLASS_MAPPINGS	["applyColorMap_3"] = cv2_applyColorMap_3

class cv2_approxPolyDP_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"curve"	: ("NPARRAY",),
				"epsilon"	: ("FLOAT",),
				"closed"	: ("BOOLEAN",),
			},
			'optional': {
				"approxCurve"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, curve, epsilon, closed, approxCurve=None):
		ret = apply_function(cv2.approxPolyDP, [curve, epsilon, closed, approxCurve], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["approxPolyDP_0"] = "OpenCV approxPolyDP_0"
NODE_CLASS_MAPPINGS	["approxPolyDP_0"] = cv2_approxPolyDP_0

class cv2_approxPolyDP_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"curve"	: ("NPARRAY",),
				"epsilon"	: ("FLOAT",),
				"closed"	: ("BOOLEAN",),
			},
			'optional': {
				"approxCurve"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, curve, epsilon, closed, approxCurve=None):
		ret = apply_function(cv2.approxPolyDP, [curve, epsilon, closed, approxCurve], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["approxPolyDP_1"] = "OpenCV approxPolyDP_1"
NODE_CLASS_MAPPINGS	["approxPolyDP_1"] = cv2_approxPolyDP_1

class cv2_approxPolyN_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"curve"	: ("NPARRAY",),
				"nsides"	: ("INT",),
				"epsilon_percentage"	: ("FLOAT",),
				"ensure_convex"	: ("BOOLEAN",),
			},
			'optional': {
				"approxCurve"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, curve, nsides, epsilon_percentage, ensure_convex, approxCurve=None):
		ret = apply_function(cv2.approxPolyN, [curve, nsides, approxCurve, epsilon_percentage, ensure_convex], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["approxPolyN_0"] = "OpenCV approxPolyN_0"
NODE_CLASS_MAPPINGS	["approxPolyN_0"] = cv2_approxPolyN_0

class cv2_approxPolyN_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"curve"	: ("NPARRAY",),
				"nsides"	: ("INT",),
				"epsilon_percentage"	: ("FLOAT",),
				"ensure_convex"	: ("BOOLEAN",),
			},
			'optional': {
				"approxCurve"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, curve, nsides, epsilon_percentage, ensure_convex, approxCurve=None):
		ret = apply_function(cv2.approxPolyN, [curve, nsides, approxCurve, epsilon_percentage, ensure_convex], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["approxPolyN_1"] = "OpenCV approxPolyN_1"
NODE_CLASS_MAPPINGS	["approxPolyN_1"] = cv2_approxPolyN_1

class cv2_arcLength_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"curve"	: ("NPARRAY",),
				"closed"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, curve, closed):
		ret = apply_function(cv2.arcLength, [curve, closed], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["arcLength_0"] = "OpenCV arcLength_0"
NODE_CLASS_MAPPINGS	["arcLength_0"] = cv2_arcLength_0

class cv2_arcLength_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"curve"	: ("NPARRAY",),
				"closed"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, curve, closed):
		ret = apply_function(cv2.arcLength, [curve, closed], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["arcLength_1"] = "OpenCV arcLength_1"
NODE_CLASS_MAPPINGS	["arcLength_1"] = cv2_arcLength_1

class cv2_arrowedLine_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"pt1"	: ("STRING",),
				"pt2"	: ("STRING",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"line_type"	: ("INT",),
				"shift"	: ("INT",),
				"tipLength"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, pt1, pt2, color, thickness, line_type, shift, tipLength):
		ret = apply_function(cv2.arrowedLine, [img, pt1, pt2, color, thickness, line_type, shift, tipLength], [0], [1, 2, 3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["arrowedLine_0"] = "OpenCV arrowedLine_0"
NODE_CLASS_MAPPINGS	["arrowedLine_0"] = cv2_arrowedLine_0

class cv2_arrowedLine_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"pt1"	: ("STRING",),
				"pt2"	: ("STRING",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"line_type"	: ("INT",),
				"shift"	: ("INT",),
				"tipLength"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, pt1, pt2, color, thickness, line_type, shift, tipLength):
		ret = apply_function(cv2.arrowedLine, [img, pt1, pt2, color, thickness, line_type, shift, tipLength], [0], [1, 2, 3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["arrowedLine_1"] = "OpenCV arrowedLine_1"
NODE_CLASS_MAPPINGS	["arrowedLine_1"] = cv2_arrowedLine_1

class cv2_batchDistance_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"dtype"	: ("INT",),
				"normType"	: ("INT",),
				"K"	: ("INT",),
				"update"	: ("INT",),
				"crosscheck"	: ("BOOLEAN",),
			},
			'optional': {
				"dist"	: ("NPARRAY",),
				"nidx"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dtype, normType, K, update, crosscheck, dist=None, nidx=None, mask=None):
		ret = apply_function(cv2.batchDistance, [src1, src2, dtype, dist, nidx, normType, K, mask, update, crosscheck], [0, 1, 3, 4, 7], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["batchDistance_0"] = "OpenCV batchDistance_0"
NODE_CLASS_MAPPINGS	["batchDistance_0"] = cv2_batchDistance_0

class cv2_batchDistance_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"dtype"	: ("INT",),
				"normType"	: ("INT",),
				"K"	: ("INT",),
				"update"	: ("INT",),
				"crosscheck"	: ("BOOLEAN",),
			},
			'optional': {
				"dist"	: ("NPARRAY",),
				"nidx"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dtype, normType, K, update, crosscheck, dist=None, nidx=None, mask=None):
		ret = apply_function(cv2.batchDistance, [src1, src2, dtype, dist, nidx, normType, K, mask, update, crosscheck], [0, 1, 3, 4, 7], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["batchDistance_1"] = "OpenCV batchDistance_1"
NODE_CLASS_MAPPINGS	["batchDistance_1"] = cv2_batchDistance_1

class cv2_bilateralFilter_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"d"	: ("INT",),
				"sigmaColor"	: ("FLOAT",),
				"sigmaSpace"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, d, sigmaColor, sigmaSpace, borderType, dst=None):
		ret = apply_function(cv2.bilateralFilter, [src, d, sigmaColor, sigmaSpace, dst, borderType], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["bilateralFilter_0"] = "OpenCV bilateralFilter_0"
NODE_CLASS_MAPPINGS	["bilateralFilter_0"] = cv2_bilateralFilter_0

class cv2_bilateralFilter_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"d"	: ("INT",),
				"sigmaColor"	: ("FLOAT",),
				"sigmaSpace"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, d, sigmaColor, sigmaSpace, borderType, dst=None):
		ret = apply_function(cv2.bilateralFilter, [src, d, sigmaColor, sigmaSpace, dst, borderType], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["bilateralFilter_1"] = "OpenCV bilateralFilter_1"
NODE_CLASS_MAPPINGS	["bilateralFilter_1"] = cv2_bilateralFilter_1

class cv2_bitwise_and_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None, mask=None):
		ret = apply_function(cv2.bitwise_and, [src1, src2, dst, mask], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["bitwise_and_0"] = "OpenCV bitwise_and_0"
NODE_CLASS_MAPPINGS	["bitwise_and_0"] = cv2_bitwise_and_0

class cv2_bitwise_and_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None, mask=None):
		ret = apply_function(cv2.bitwise_and, [src1, src2, dst, mask], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["bitwise_and_1"] = "OpenCV bitwise_and_1"
NODE_CLASS_MAPPINGS	["bitwise_and_1"] = cv2_bitwise_and_1

class cv2_bitwise_not_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None, mask=None):
		ret = apply_function(cv2.bitwise_not, [src, dst, mask], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["bitwise_not_0"] = "OpenCV bitwise_not_0"
NODE_CLASS_MAPPINGS	["bitwise_not_0"] = cv2_bitwise_not_0

class cv2_bitwise_not_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None, mask=None):
		ret = apply_function(cv2.bitwise_not, [src, dst, mask], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["bitwise_not_1"] = "OpenCV bitwise_not_1"
NODE_CLASS_MAPPINGS	["bitwise_not_1"] = cv2_bitwise_not_1

class cv2_bitwise_or_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None, mask=None):
		ret = apply_function(cv2.bitwise_or, [src1, src2, dst, mask], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["bitwise_or_0"] = "OpenCV bitwise_or_0"
NODE_CLASS_MAPPINGS	["bitwise_or_0"] = cv2_bitwise_or_0

class cv2_bitwise_or_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None, mask=None):
		ret = apply_function(cv2.bitwise_or, [src1, src2, dst, mask], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["bitwise_or_1"] = "OpenCV bitwise_or_1"
NODE_CLASS_MAPPINGS	["bitwise_or_1"] = cv2_bitwise_or_1

class cv2_bitwise_xor_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None, mask=None):
		ret = apply_function(cv2.bitwise_xor, [src1, src2, dst, mask], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["bitwise_xor_0"] = "OpenCV bitwise_xor_0"
NODE_CLASS_MAPPINGS	["bitwise_xor_0"] = cv2_bitwise_xor_0

class cv2_bitwise_xor_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None, mask=None):
		ret = apply_function(cv2.bitwise_xor, [src1, src2, dst, mask], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["bitwise_xor_1"] = "OpenCV bitwise_xor_1"
NODE_CLASS_MAPPINGS	["bitwise_xor_1"] = cv2_bitwise_xor_1

class cv2_blendLinear_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"weights1"	: ("NPARRAY",),
				"weights2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, weights1, weights2, dst=None):
		ret = apply_function(cv2.blendLinear, [src1, src2, weights1, weights2, dst], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["blendLinear_0"] = "OpenCV blendLinear_0"
NODE_CLASS_MAPPINGS	["blendLinear_0"] = cv2_blendLinear_0

class cv2_blendLinear_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"weights1"	: ("NPARRAY",),
				"weights2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, weights1, weights2, dst=None):
		ret = apply_function(cv2.blendLinear, [src1, src2, weights1, weights2, dst], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["blendLinear_1"] = "OpenCV blendLinear_1"
NODE_CLASS_MAPPINGS	["blendLinear_1"] = cv2_blendLinear_1

class cv2_blur_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("STRING",),
				"anchor"	: ("STRING",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, anchor, borderType, dst=None):
		ret = apply_function(cv2.blur, [src, ksize, dst, anchor, borderType], [0, 2], [1, 3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["blur_0"] = "OpenCV blur_0"
NODE_CLASS_MAPPINGS	["blur_0"] = cv2_blur_0

class cv2_blur_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("STRING",),
				"anchor"	: ("STRING",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, anchor, borderType, dst=None):
		ret = apply_function(cv2.blur, [src, ksize, dst, anchor, borderType], [0, 2], [1, 3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["blur_1"] = "OpenCV blur_1"
NODE_CLASS_MAPPINGS	["blur_1"] = cv2_blur_1

class cv2_borderInterpolate_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"p"	: ("INT",),
				"len"	: ("INT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, p, len, borderType):
		ret = apply_function(cv2.borderInterpolate, [p, len, borderType], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["borderInterpolate_0"] = "OpenCV borderInterpolate_0"
NODE_CLASS_MAPPINGS	["borderInterpolate_0"] = cv2_borderInterpolate_0

class cv2_boundingRect_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"array"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, array):
		ret = apply_function(cv2.boundingRect, [array], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["boundingRect_0"] = "OpenCV boundingRect_0"
NODE_CLASS_MAPPINGS	["boundingRect_0"] = cv2_boundingRect_0

class cv2_boundingRect_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"array"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, array):
		ret = apply_function(cv2.boundingRect, [array], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["boundingRect_1"] = "OpenCV boundingRect_1"
NODE_CLASS_MAPPINGS	["boundingRect_1"] = cv2_boundingRect_1

class cv2_boxFilter_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"ksize"	: ("STRING",),
				"anchor"	: ("STRING",),
				"normalize"	: ("BOOLEAN",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, ksize, anchor, normalize, borderType, dst=None):
		ret = apply_function(cv2.boxFilter, [src, ddepth, ksize, dst, anchor, normalize, borderType], [0, 3], [2, 4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["boxFilter_0"] = "OpenCV boxFilter_0"
NODE_CLASS_MAPPINGS	["boxFilter_0"] = cv2_boxFilter_0

class cv2_boxFilter_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"ksize"	: ("STRING",),
				"anchor"	: ("STRING",),
				"normalize"	: ("BOOLEAN",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, ksize, anchor, normalize, borderType, dst=None):
		ret = apply_function(cv2.boxFilter, [src, ddepth, ksize, dst, anchor, normalize, borderType], [0, 3], [2, 4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["boxFilter_1"] = "OpenCV boxFilter_1"
NODE_CLASS_MAPPINGS	["boxFilter_1"] = cv2_boxFilter_1

class cv2_boxPoints_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"box"	: ("STRING",),
			},
			'optional': {
				"points"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, box, points=None):
		ret = apply_function(cv2.boxPoints, [box, points], [1], [0])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["boxPoints_0"] = "OpenCV boxPoints_0"
NODE_CLASS_MAPPINGS	["boxPoints_0"] = cv2_boxPoints_0

class cv2_boxPoints_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"box"	: ("STRING",),
			},
			'optional': {
				"points"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, box, points=None):
		ret = apply_function(cv2.boxPoints, [box, points], [1], [0])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["boxPoints_1"] = "OpenCV boxPoints_1"
NODE_CLASS_MAPPINGS	["boxPoints_1"] = cv2_boxPoints_1

class cv2_broadcast_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"shape"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, shape, dst=None):
		ret = apply_function(cv2.broadcast, [src, shape, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["broadcast_0"] = "OpenCV broadcast_0"
NODE_CLASS_MAPPINGS	["broadcast_0"] = cv2_broadcast_0

class cv2_broadcast_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"shape"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, shape, dst=None):
		ret = apply_function(cv2.broadcast, [src, shape, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["broadcast_1"] = "OpenCV broadcast_1"
NODE_CLASS_MAPPINGS	["broadcast_1"] = cv2_broadcast_1

class cv2_calcCovarMatrix_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"samples"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"ctype"	: ("INT",),
			},
			'optional': {
				"covar"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, samples, mean, flags, ctype, covar=None):
		ret = apply_function(cv2.calcCovarMatrix, [samples, mean, flags, covar, ctype], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["calcCovarMatrix_0"] = "OpenCV calcCovarMatrix_0"
NODE_CLASS_MAPPINGS	["calcCovarMatrix_0"] = cv2_calcCovarMatrix_0

class cv2_calcCovarMatrix_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"samples"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"ctype"	: ("INT",),
			},
			'optional': {
				"covar"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, samples, mean, flags, ctype, covar=None):
		ret = apply_function(cv2.calcCovarMatrix, [samples, mean, flags, covar, ctype], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["calcCovarMatrix_1"] = "OpenCV calcCovarMatrix_1"
NODE_CLASS_MAPPINGS	["calcCovarMatrix_1"] = cv2_calcCovarMatrix_1

class cv2_calcOpticalFlowFarneback_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"prev"	: ("NPARRAY",),
				"next"	: ("NPARRAY",),
				"flow"	: ("NPARRAY",),
				"pyr_scale"	: ("FLOAT",),
				"levels"	: ("INT",),
				"winsize"	: ("INT",),
				"iterations"	: ("INT",),
				"poly_n"	: ("INT",),
				"poly_sigma"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags):
		ret = apply_function(cv2.calcOpticalFlowFarneback, [prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["calcOpticalFlowFarneback_0"] = "OpenCV calcOpticalFlowFarneback_0"
NODE_CLASS_MAPPINGS	["calcOpticalFlowFarneback_0"] = cv2_calcOpticalFlowFarneback_0

class cv2_calcOpticalFlowFarneback_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"prev"	: ("NPARRAY",),
				"next"	: ("NPARRAY",),
				"flow"	: ("NPARRAY",),
				"pyr_scale"	: ("FLOAT",),
				"levels"	: ("INT",),
				"winsize"	: ("INT",),
				"iterations"	: ("INT",),
				"poly_n"	: ("INT",),
				"poly_sigma"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags):
		ret = apply_function(cv2.calcOpticalFlowFarneback, [prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["calcOpticalFlowFarneback_1"] = "OpenCV calcOpticalFlowFarneback_1"
NODE_CLASS_MAPPINGS	["calcOpticalFlowFarneback_1"] = cv2_calcOpticalFlowFarneback_1

class cv2_calcOpticalFlowPyrLK_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"prevImg"	: ("NPARRAY",),
				"nextImg"	: ("NPARRAY",),
				"prevPts"	: ("NPARRAY",),
				"nextPts"	: ("NPARRAY",),
				"winSize"	: ("STRING",),
				"maxLevel"	: ("INT",),
				"criteria"	: ("STRING",),
				"flags"	: ("INT",),
				"minEigThreshold"	: ("FLOAT",),
			},
			'optional': {
				"status"	: ("NPARRAY",),
				"err"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, prevImg, nextImg, prevPts, nextPts, winSize, maxLevel, criteria, flags, minEigThreshold, status=None, err=None):
		ret = apply_function(cv2.calcOpticalFlowPyrLK, [prevImg, nextImg, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold], [0, 1, 2, 3, 4, 5], [6, 8])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["calcOpticalFlowPyrLK_0"] = "OpenCV calcOpticalFlowPyrLK_0"
NODE_CLASS_MAPPINGS	["calcOpticalFlowPyrLK_0"] = cv2_calcOpticalFlowPyrLK_0

class cv2_calcOpticalFlowPyrLK_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"prevImg"	: ("NPARRAY",),
				"nextImg"	: ("NPARRAY",),
				"prevPts"	: ("NPARRAY",),
				"nextPts"	: ("NPARRAY",),
				"winSize"	: ("STRING",),
				"maxLevel"	: ("INT",),
				"criteria"	: ("STRING",),
				"flags"	: ("INT",),
				"minEigThreshold"	: ("FLOAT",),
			},
			'optional': {
				"status"	: ("NPARRAY",),
				"err"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, prevImg, nextImg, prevPts, nextPts, winSize, maxLevel, criteria, flags, minEigThreshold, status=None, err=None):
		ret = apply_function(cv2.calcOpticalFlowPyrLK, [prevImg, nextImg, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold], [0, 1, 2, 3, 4, 5], [6, 8])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["calcOpticalFlowPyrLK_1"] = "OpenCV calcOpticalFlowPyrLK_1"
NODE_CLASS_MAPPINGS	["calcOpticalFlowPyrLK_1"] = cv2_calcOpticalFlowPyrLK_1

class cv2_calibrationMatrixValues_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix"	: ("NPARRAY",),
				"imageSize"	: ("STRING",),
				"apertureWidth"	: ("FLOAT",),
				"apertureHeight"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT", "FLOAT", "FLOAT", "STRING", "FLOAT",)
	RETURN_NAMES	= ("float_0", "float_1", "float_2", "literal", "float_4",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix, imageSize, apertureWidth, apertureHeight):
		ret = apply_function(cv2.calibrationMatrixValues, [cameraMatrix, imageSize, apertureWidth, apertureHeight], [0], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["calibrationMatrixValues_0"] = "OpenCV calibrationMatrixValues_0"
NODE_CLASS_MAPPINGS	["calibrationMatrixValues_0"] = cv2_calibrationMatrixValues_0

class cv2_calibrationMatrixValues_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix"	: ("NPARRAY",),
				"imageSize"	: ("STRING",),
				"apertureWidth"	: ("FLOAT",),
				"apertureHeight"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT", "FLOAT", "FLOAT", "STRING", "FLOAT",)
	RETURN_NAMES	= ("float_0", "float_1", "float_2", "literal", "float_4",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix, imageSize, apertureWidth, apertureHeight):
		ret = apply_function(cv2.calibrationMatrixValues, [cameraMatrix, imageSize, apertureWidth, apertureHeight], [0], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["calibrationMatrixValues_1"] = "OpenCV calibrationMatrixValues_1"
NODE_CLASS_MAPPINGS	["calibrationMatrixValues_1"] = cv2_calibrationMatrixValues_1

class cv2_cartToPolar_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"x"	: ("NPARRAY",),
				"y"	: ("NPARRAY",),
				"angleInDegrees"	: ("BOOLEAN",),
			},
			'optional': {
				"magnitude"	: ("NPARRAY",),
				"angle"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, x, y, angleInDegrees, magnitude=None, angle=None):
		ret = apply_function(cv2.cartToPolar, [x, y, magnitude, angle, angleInDegrees], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cartToPolar_0"] = "OpenCV cartToPolar_0"
NODE_CLASS_MAPPINGS	["cartToPolar_0"] = cv2_cartToPolar_0

class cv2_cartToPolar_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"x"	: ("NPARRAY",),
				"y"	: ("NPARRAY",),
				"angleInDegrees"	: ("BOOLEAN",),
			},
			'optional': {
				"magnitude"	: ("NPARRAY",),
				"angle"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, x, y, angleInDegrees, magnitude=None, angle=None):
		ret = apply_function(cv2.cartToPolar, [x, y, magnitude, angle, angleInDegrees], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cartToPolar_1"] = "OpenCV cartToPolar_1"
NODE_CLASS_MAPPINGS	["cartToPolar_1"] = cv2_cartToPolar_1

class cv2_checkChessboard_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"size"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, size):
		ret = apply_function(cv2.checkChessboard, [img, size], [0], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["checkChessboard_0"] = "OpenCV checkChessboard_0"
NODE_CLASS_MAPPINGS	["checkChessboard_0"] = cv2_checkChessboard_0

class cv2_checkChessboard_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"size"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, size):
		ret = apply_function(cv2.checkChessboard, [img, size], [0], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["checkChessboard_1"] = "OpenCV checkChessboard_1"
NODE_CLASS_MAPPINGS	["checkChessboard_1"] = cv2_checkChessboard_1

class cv2_checkHardwareSupport_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"feature"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, feature):
		ret = apply_function(cv2.checkHardwareSupport, [feature], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["checkHardwareSupport_0"] = "OpenCV checkHardwareSupport_0"
NODE_CLASS_MAPPINGS	["checkHardwareSupport_0"] = cv2_checkHardwareSupport_0

class cv2_checkRange_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"a"	: ("NPARRAY",),
				"quiet"	: ("BOOLEAN",),
				"minVal"	: ("FLOAT",),
				"maxVal"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "STRING",)
	RETURN_NAMES	= ("bool", "literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, a, quiet, minVal, maxVal):
		ret = apply_function(cv2.checkRange, [a, quiet, minVal, maxVal], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["checkRange_0"] = "OpenCV checkRange_0"
NODE_CLASS_MAPPINGS	["checkRange_0"] = cv2_checkRange_0

class cv2_checkRange_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"a"	: ("NPARRAY",),
				"quiet"	: ("BOOLEAN",),
				"minVal"	: ("FLOAT",),
				"maxVal"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "STRING",)
	RETURN_NAMES	= ("bool", "literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, a, quiet, minVal, maxVal):
		ret = apply_function(cv2.checkRange, [a, quiet, minVal, maxVal], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["checkRange_1"] = "OpenCV checkRange_1"
NODE_CLASS_MAPPINGS	["checkRange_1"] = cv2_checkRange_1

class cv2_circle_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"center"	: ("STRING",),
				"radius"	: ("INT",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, center, radius, color, thickness, lineType, shift):
		ret = apply_function(cv2.circle, [img, center, radius, color, thickness, lineType, shift], [0], [1, 3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["circle_0"] = "OpenCV circle_0"
NODE_CLASS_MAPPINGS	["circle_0"] = cv2_circle_0

class cv2_circle_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"center"	: ("STRING",),
				"radius"	: ("INT",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, center, radius, color, thickness, lineType, shift):
		ret = apply_function(cv2.circle, [img, center, radius, color, thickness, lineType, shift], [0], [1, 3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["circle_1"] = "OpenCV circle_1"
NODE_CLASS_MAPPINGS	["circle_1"] = cv2_circle_1

class cv2_clipLine_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"imgRect"	: ("STRING",),
				"pt1"	: ("STRING",),
				"pt2"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "STRING", "STRING",)
	RETURN_NAMES	= ("bool", "literal_1", "literal_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, imgRect, pt1, pt2):
		ret = apply_function(cv2.clipLine, [imgRect, pt1, pt2], [], [0, 1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["clipLine_0"] = "OpenCV clipLine_0"
NODE_CLASS_MAPPINGS	["clipLine_0"] = cv2_clipLine_0

class cv2_colorChange_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"red_mul"	: ("FLOAT",),
				"green_mul"	: ("FLOAT",),
				"blue_mul"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask, red_mul, green_mul, blue_mul, dst=None):
		ret = apply_function(cv2.colorChange, [src, mask, dst, red_mul, green_mul, blue_mul], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["colorChange_0"] = "OpenCV colorChange_0"
NODE_CLASS_MAPPINGS	["colorChange_0"] = cv2_colorChange_0

class cv2_colorChange_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"red_mul"	: ("FLOAT",),
				"green_mul"	: ("FLOAT",),
				"blue_mul"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask, red_mul, green_mul, blue_mul, dst=None):
		ret = apply_function(cv2.colorChange, [src, mask, dst, red_mul, green_mul, blue_mul], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["colorChange_1"] = "OpenCV colorChange_1"
NODE_CLASS_MAPPINGS	["colorChange_1"] = cv2_colorChange_1

class cv2_compare_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"cmpop"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, cmpop, dst=None):
		ret = apply_function(cv2.compare, [src1, src2, cmpop, dst], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["compare_0"] = "OpenCV compare_0"
NODE_CLASS_MAPPINGS	["compare_0"] = cv2_compare_0

class cv2_compare_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"cmpop"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, cmpop, dst=None):
		ret = apply_function(cv2.compare, [src1, src2, cmpop, dst], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["compare_1"] = "OpenCV compare_1"
NODE_CLASS_MAPPINGS	["compare_1"] = cv2_compare_1

class cv2_compareHist_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"H1"	: ("NPARRAY",),
				"H2"	: ("NPARRAY",),
				"method"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, H1, H2, method):
		ret = apply_function(cv2.compareHist, [H1, H2, method], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["compareHist_0"] = "OpenCV compareHist_0"
NODE_CLASS_MAPPINGS	["compareHist_0"] = cv2_compareHist_0

class cv2_compareHist_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"H1"	: ("NPARRAY",),
				"H2"	: ("NPARRAY",),
				"method"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, H1, H2, method):
		ret = apply_function(cv2.compareHist, [H1, H2, method], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["compareHist_1"] = "OpenCV compareHist_1"
NODE_CLASS_MAPPINGS	["compareHist_1"] = cv2_compareHist_1

class cv2_completeSymm_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"m"	: ("NPARRAY",),
				"lowerToUpper"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, m, lowerToUpper):
		ret = apply_function(cv2.completeSymm, [m, lowerToUpper], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["completeSymm_0"] = "OpenCV completeSymm_0"
NODE_CLASS_MAPPINGS	["completeSymm_0"] = cv2_completeSymm_0

class cv2_completeSymm_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"m"	: ("NPARRAY",),
				"lowerToUpper"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, m, lowerToUpper):
		ret = apply_function(cv2.completeSymm, [m, lowerToUpper], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["completeSymm_1"] = "OpenCV completeSymm_1"
NODE_CLASS_MAPPINGS	["completeSymm_1"] = cv2_completeSymm_1

class cv2_composeRT_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"rvec1"	: ("NPARRAY",),
				"tvec1"	: ("NPARRAY",),
				"rvec2"	: ("NPARRAY",),
				"tvec2"	: ("NPARRAY",),
			},
			'optional': {
				"rvec3"	: ("NPARRAY",),
				"tvec3"	: ("NPARRAY",),
				"dr3dr1"	: ("NPARRAY",),
				"dr3dt1"	: ("NPARRAY",),
				"dr3dr2"	: ("NPARRAY",),
				"dr3dt2"	: ("NPARRAY",),
				"dt3dr1"	: ("NPARRAY",),
				"dt3dt1"	: ("NPARRAY",),
				"dt3dr2"	: ("NPARRAY",),
				"dt3dt2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2", "nparray_3", "nparray_4", "nparray_5", "nparray_6", "nparray_7", "nparray_8", "nparray_9",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, rvec1, tvec1, rvec2, tvec2, rvec3=None, tvec3=None, dr3dr1=None, dr3dt1=None, dr3dr2=None, dr3dt2=None, dt3dr1=None, dt3dt1=None, dt3dr2=None, dt3dt2=None):
		ret = apply_function(cv2.composeRT, [rvec1, tvec1, rvec2, tvec2, rvec3, tvec3, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["composeRT_0"] = "OpenCV composeRT_0"
NODE_CLASS_MAPPINGS	["composeRT_0"] = cv2_composeRT_0

class cv2_composeRT_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"rvec1"	: ("NPARRAY",),
				"tvec1"	: ("NPARRAY",),
				"rvec2"	: ("NPARRAY",),
				"tvec2"	: ("NPARRAY",),
			},
			'optional': {
				"rvec3"	: ("NPARRAY",),
				"tvec3"	: ("NPARRAY",),
				"dr3dr1"	: ("NPARRAY",),
				"dr3dt1"	: ("NPARRAY",),
				"dr3dr2"	: ("NPARRAY",),
				"dr3dt2"	: ("NPARRAY",),
				"dt3dr1"	: ("NPARRAY",),
				"dt3dt1"	: ("NPARRAY",),
				"dt3dr2"	: ("NPARRAY",),
				"dt3dt2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2", "nparray_3", "nparray_4", "nparray_5", "nparray_6", "nparray_7", "nparray_8", "nparray_9",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, rvec1, tvec1, rvec2, tvec2, rvec3=None, tvec3=None, dr3dr1=None, dr3dt1=None, dr3dr2=None, dr3dt2=None, dt3dr1=None, dt3dt1=None, dt3dr2=None, dt3dt2=None):
		ret = apply_function(cv2.composeRT, [rvec1, tvec1, rvec2, tvec2, rvec3, tvec3, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["composeRT_1"] = "OpenCV composeRT_1"
NODE_CLASS_MAPPINGS	["composeRT_1"] = cv2_composeRT_1

class cv2_computeCorrespondEpilines_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
				"whichImage"	: ("INT",),
				"F"	: ("NPARRAY",),
			},
			'optional': {
				"lines"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points, whichImage, F, lines=None):
		ret = apply_function(cv2.computeCorrespondEpilines, [points, whichImage, F, lines], [0, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["computeCorrespondEpilines_0"] = "OpenCV computeCorrespondEpilines_0"
NODE_CLASS_MAPPINGS	["computeCorrespondEpilines_0"] = cv2_computeCorrespondEpilines_0

class cv2_computeCorrespondEpilines_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
				"whichImage"	: ("INT",),
				"F"	: ("NPARRAY",),
			},
			'optional': {
				"lines"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points, whichImage, F, lines=None):
		ret = apply_function(cv2.computeCorrespondEpilines, [points, whichImage, F, lines], [0, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["computeCorrespondEpilines_1"] = "OpenCV computeCorrespondEpilines_1"
NODE_CLASS_MAPPINGS	["computeCorrespondEpilines_1"] = cv2_computeCorrespondEpilines_1

class cv2_computeECC_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"templateImage"	: ("NPARRAY",),
				"inputImage"	: ("NPARRAY",),
			},
			'optional': {
				"inputMask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, templateImage, inputImage, inputMask=None):
		ret = apply_function(cv2.computeECC, [templateImage, inputImage, inputMask], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["computeECC_0"] = "OpenCV computeECC_0"
NODE_CLASS_MAPPINGS	["computeECC_0"] = cv2_computeECC_0

class cv2_computeECC_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"templateImage"	: ("NPARRAY",),
				"inputImage"	: ("NPARRAY",),
			},
			'optional': {
				"inputMask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, templateImage, inputImage, inputMask=None):
		ret = apply_function(cv2.computeECC, [templateImage, inputImage, inputMask], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["computeECC_1"] = "OpenCV computeECC_1"
NODE_CLASS_MAPPINGS	["computeECC_1"] = cv2_computeECC_1

class cv2_connectedComponents_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"connectivity"	: ("INT",),
				"ltype"	: ("INT",),
			},
			'optional': {
				"labels"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, connectivity, ltype, labels=None):
		ret = apply_function(cv2.connectedComponents, [image, labels, connectivity, ltype], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["connectedComponents_0"] = "OpenCV connectedComponents_0"
NODE_CLASS_MAPPINGS	["connectedComponents_0"] = cv2_connectedComponents_0

class cv2_connectedComponents_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"connectivity"	: ("INT",),
				"ltype"	: ("INT",),
			},
			'optional': {
				"labels"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, connectivity, ltype, labels=None):
		ret = apply_function(cv2.connectedComponents, [image, labels, connectivity, ltype], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["connectedComponents_1"] = "OpenCV connectedComponents_1"
NODE_CLASS_MAPPINGS	["connectedComponents_1"] = cv2_connectedComponents_1

class cv2_connectedComponentsWithAlgorithm_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"connectivity"	: ("INT",),
				"ltype"	: ("INT",),
				"ccltype"	: ("INT",),
			},
			'optional': {
				"labels"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, connectivity, ltype, ccltype, labels=None):
		ret = apply_function(cv2.connectedComponentsWithAlgorithm, [image, connectivity, ltype, ccltype, labels], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["connectedComponentsWithAlgorithm_0"] = "OpenCV connectedComponentsWithAlgorithm_0"
NODE_CLASS_MAPPINGS	["connectedComponentsWithAlgorithm_0"] = cv2_connectedComponentsWithAlgorithm_0

class cv2_connectedComponentsWithAlgorithm_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"connectivity"	: ("INT",),
				"ltype"	: ("INT",),
				"ccltype"	: ("INT",),
			},
			'optional': {
				"labels"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, connectivity, ltype, ccltype, labels=None):
		ret = apply_function(cv2.connectedComponentsWithAlgorithm, [image, connectivity, ltype, ccltype, labels], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["connectedComponentsWithAlgorithm_1"] = "OpenCV connectedComponentsWithAlgorithm_1"
NODE_CLASS_MAPPINGS	["connectedComponentsWithAlgorithm_1"] = cv2_connectedComponentsWithAlgorithm_1

class cv2_connectedComponentsWithStats_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"connectivity"	: ("INT",),
				"ltype"	: ("INT",),
			},
			'optional': {
				"labels"	: ("NPARRAY",),
				"stats"	: ("NPARRAY",),
				"centroids"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, connectivity, ltype, labels=None, stats=None, centroids=None):
		ret = apply_function(cv2.connectedComponentsWithStats, [image, labels, stats, centroids, connectivity, ltype], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["connectedComponentsWithStats_0"] = "OpenCV connectedComponentsWithStats_0"
NODE_CLASS_MAPPINGS	["connectedComponentsWithStats_0"] = cv2_connectedComponentsWithStats_0

class cv2_connectedComponentsWithStats_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"connectivity"	: ("INT",),
				"ltype"	: ("INT",),
			},
			'optional': {
				"labels"	: ("NPARRAY",),
				"stats"	: ("NPARRAY",),
				"centroids"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, connectivity, ltype, labels=None, stats=None, centroids=None):
		ret = apply_function(cv2.connectedComponentsWithStats, [image, labels, stats, centroids, connectivity, ltype], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["connectedComponentsWithStats_1"] = "OpenCV connectedComponentsWithStats_1"
NODE_CLASS_MAPPINGS	["connectedComponentsWithStats_1"] = cv2_connectedComponentsWithStats_1

class cv2_connectedComponentsWithStatsWithAlgorithm_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"connectivity"	: ("INT",),
				"ltype"	: ("INT",),
				"ccltype"	: ("INT",),
			},
			'optional': {
				"labels"	: ("NPARRAY",),
				"stats"	: ("NPARRAY",),
				"centroids"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, connectivity, ltype, ccltype, labels=None, stats=None, centroids=None):
		ret = apply_function(cv2.connectedComponentsWithStatsWithAlgorithm, [image, connectivity, ltype, ccltype, labels, stats, centroids], [0, 4, 5, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["connectedComponentsWithStatsWithAlgorithm_0"] = "OpenCV connectedComponentsWithStatsWithAlgorithm_0"
NODE_CLASS_MAPPINGS	["connectedComponentsWithStatsWithAlgorithm_0"] = cv2_connectedComponentsWithStatsWithAlgorithm_0

class cv2_connectedComponentsWithStatsWithAlgorithm_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"connectivity"	: ("INT",),
				"ltype"	: ("INT",),
				"ccltype"	: ("INT",),
			},
			'optional': {
				"labels"	: ("NPARRAY",),
				"stats"	: ("NPARRAY",),
				"centroids"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, connectivity, ltype, ccltype, labels=None, stats=None, centroids=None):
		ret = apply_function(cv2.connectedComponentsWithStatsWithAlgorithm, [image, connectivity, ltype, ccltype, labels, stats, centroids], [0, 4, 5, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["connectedComponentsWithStatsWithAlgorithm_1"] = "OpenCV connectedComponentsWithStatsWithAlgorithm_1"
NODE_CLASS_MAPPINGS	["connectedComponentsWithStatsWithAlgorithm_1"] = cv2_connectedComponentsWithStatsWithAlgorithm_1

class cv2_contourArea_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"contour"	: ("NPARRAY",),
				"oriented"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, contour, oriented):
		ret = apply_function(cv2.contourArea, [contour, oriented], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["contourArea_0"] = "OpenCV contourArea_0"
NODE_CLASS_MAPPINGS	["contourArea_0"] = cv2_contourArea_0

class cv2_contourArea_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"contour"	: ("NPARRAY",),
				"oriented"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, contour, oriented):
		ret = apply_function(cv2.contourArea, [contour, oriented], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["contourArea_1"] = "OpenCV contourArea_1"
NODE_CLASS_MAPPINGS	["contourArea_1"] = cv2_contourArea_1

class cv2_convertFp16_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.convertFp16, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convertFp16_0"] = "OpenCV convertFp16_0"
NODE_CLASS_MAPPINGS	["convertFp16_0"] = cv2_convertFp16_0

class cv2_convertFp16_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.convertFp16, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convertFp16_1"] = "OpenCV convertFp16_1"
NODE_CLASS_MAPPINGS	["convertFp16_1"] = cv2_convertFp16_1

class cv2_convertMaps_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"map1"	: ("NPARRAY",),
				"map2"	: ("NPARRAY",),
				"dstmap1type"	: ("INT",),
				"nninterpolation"	: ("BOOLEAN",),
			},
			'optional': {
				"dstmap1"	: ("NPARRAY",),
				"dstmap2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, map1, map2, dstmap1type, nninterpolation, dstmap1=None, dstmap2=None):
		ret = apply_function(cv2.convertMaps, [map1, map2, dstmap1type, dstmap1, dstmap2, nninterpolation], [0, 1, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convertMaps_0"] = "OpenCV convertMaps_0"
NODE_CLASS_MAPPINGS	["convertMaps_0"] = cv2_convertMaps_0

class cv2_convertMaps_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"map1"	: ("NPARRAY",),
				"map2"	: ("NPARRAY",),
				"dstmap1type"	: ("INT",),
				"nninterpolation"	: ("BOOLEAN",),
			},
			'optional': {
				"dstmap1"	: ("NPARRAY",),
				"dstmap2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, map1, map2, dstmap1type, nninterpolation, dstmap1=None, dstmap2=None):
		ret = apply_function(cv2.convertMaps, [map1, map2, dstmap1type, dstmap1, dstmap2, nninterpolation], [0, 1, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convertMaps_1"] = "OpenCV convertMaps_1"
NODE_CLASS_MAPPINGS	["convertMaps_1"] = cv2_convertMaps_1

class cv2_convertPointsFromHomogeneous_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.convertPointsFromHomogeneous, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convertPointsFromHomogeneous_0"] = "OpenCV convertPointsFromHomogeneous_0"
NODE_CLASS_MAPPINGS	["convertPointsFromHomogeneous_0"] = cv2_convertPointsFromHomogeneous_0

class cv2_convertPointsFromHomogeneous_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.convertPointsFromHomogeneous, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convertPointsFromHomogeneous_1"] = "OpenCV convertPointsFromHomogeneous_1"
NODE_CLASS_MAPPINGS	["convertPointsFromHomogeneous_1"] = cv2_convertPointsFromHomogeneous_1

class cv2_convertPointsToHomogeneous_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.convertPointsToHomogeneous, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convertPointsToHomogeneous_0"] = "OpenCV convertPointsToHomogeneous_0"
NODE_CLASS_MAPPINGS	["convertPointsToHomogeneous_0"] = cv2_convertPointsToHomogeneous_0

class cv2_convertPointsToHomogeneous_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.convertPointsToHomogeneous, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convertPointsToHomogeneous_1"] = "OpenCV convertPointsToHomogeneous_1"
NODE_CLASS_MAPPINGS	["convertPointsToHomogeneous_1"] = cv2_convertPointsToHomogeneous_1

class cv2_convertScaleAbs_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"beta"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, alpha, beta, dst=None):
		ret = apply_function(cv2.convertScaleAbs, [src, dst, alpha, beta], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convertScaleAbs_0"] = "OpenCV convertScaleAbs_0"
NODE_CLASS_MAPPINGS	["convertScaleAbs_0"] = cv2_convertScaleAbs_0

class cv2_convertScaleAbs_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"beta"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, alpha, beta, dst=None):
		ret = apply_function(cv2.convertScaleAbs, [src, dst, alpha, beta], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convertScaleAbs_1"] = "OpenCV convertScaleAbs_1"
NODE_CLASS_MAPPINGS	["convertScaleAbs_1"] = cv2_convertScaleAbs_1

class cv2_convexHull_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
				"clockwise"	: ("BOOLEAN",),
				"returnPoints"	: ("BOOLEAN",),
			},
			'optional': {
				"hull"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points, clockwise, returnPoints, hull=None):
		ret = apply_function(cv2.convexHull, [points, hull, clockwise, returnPoints], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convexHull_0"] = "OpenCV convexHull_0"
NODE_CLASS_MAPPINGS	["convexHull_0"] = cv2_convexHull_0

class cv2_convexHull_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
				"clockwise"	: ("BOOLEAN",),
				"returnPoints"	: ("BOOLEAN",),
			},
			'optional': {
				"hull"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points, clockwise, returnPoints, hull=None):
		ret = apply_function(cv2.convexHull, [points, hull, clockwise, returnPoints], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convexHull_1"] = "OpenCV convexHull_1"
NODE_CLASS_MAPPINGS	["convexHull_1"] = cv2_convexHull_1

class cv2_convexityDefects_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"contour"	: ("NPARRAY",),
				"convexhull"	: ("NPARRAY",),
			},
			'optional': {
				"convexityDefects"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, contour, convexhull, convexityDefects=None):
		ret = apply_function(cv2.convexityDefects, [contour, convexhull, convexityDefects], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convexityDefects_0"] = "OpenCV convexityDefects_0"
NODE_CLASS_MAPPINGS	["convexityDefects_0"] = cv2_convexityDefects_0

class cv2_convexityDefects_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"contour"	: ("NPARRAY",),
				"convexhull"	: ("NPARRAY",),
			},
			'optional': {
				"convexityDefects"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, contour, convexhull, convexityDefects=None):
		ret = apply_function(cv2.convexityDefects, [contour, convexhull, convexityDefects], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["convexityDefects_1"] = "OpenCV convexityDefects_1"
NODE_CLASS_MAPPINGS	["convexityDefects_1"] = cv2_convexityDefects_1

class cv2_copyMakeBorder_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"top"	: ("INT",),
				"bottom"	: ("INT",),
				"left"	: ("INT",),
				"right"	: ("INT",),
				"borderType"	: ("INT",),
				"value"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, top, bottom, left, right, borderType, value, dst=None):
		ret = apply_function(cv2.copyMakeBorder, [src, top, bottom, left, right, borderType, dst, value], [0, 6], [7])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["copyMakeBorder_0"] = "OpenCV copyMakeBorder_0"
NODE_CLASS_MAPPINGS	["copyMakeBorder_0"] = cv2_copyMakeBorder_0

class cv2_copyMakeBorder_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"top"	: ("INT",),
				"bottom"	: ("INT",),
				"left"	: ("INT",),
				"right"	: ("INT",),
				"borderType"	: ("INT",),
				"value"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, top, bottom, left, right, borderType, value, dst=None):
		ret = apply_function(cv2.copyMakeBorder, [src, top, bottom, left, right, borderType, dst, value], [0, 6], [7])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["copyMakeBorder_1"] = "OpenCV copyMakeBorder_1"
NODE_CLASS_MAPPINGS	["copyMakeBorder_1"] = cv2_copyMakeBorder_1

class cv2_copyTo_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask, dst=None):
		ret = apply_function(cv2.copyTo, [src, mask, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["copyTo_0"] = "OpenCV copyTo_0"
NODE_CLASS_MAPPINGS	["copyTo_0"] = cv2_copyTo_0

class cv2_copyTo_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask, dst=None):
		ret = apply_function(cv2.copyTo, [src, mask, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["copyTo_1"] = "OpenCV copyTo_1"
NODE_CLASS_MAPPINGS	["copyTo_1"] = cv2_copyTo_1

class cv2_cornerEigenValsAndVecs_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"blockSize"	: ("INT",),
				"ksize"	: ("INT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, blockSize, ksize, borderType, dst=None):
		ret = apply_function(cv2.cornerEigenValsAndVecs, [src, blockSize, ksize, dst, borderType], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cornerEigenValsAndVecs_0"] = "OpenCV cornerEigenValsAndVecs_0"
NODE_CLASS_MAPPINGS	["cornerEigenValsAndVecs_0"] = cv2_cornerEigenValsAndVecs_0

class cv2_cornerEigenValsAndVecs_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"blockSize"	: ("INT",),
				"ksize"	: ("INT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, blockSize, ksize, borderType, dst=None):
		ret = apply_function(cv2.cornerEigenValsAndVecs, [src, blockSize, ksize, dst, borderType], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cornerEigenValsAndVecs_1"] = "OpenCV cornerEigenValsAndVecs_1"
NODE_CLASS_MAPPINGS	["cornerEigenValsAndVecs_1"] = cv2_cornerEigenValsAndVecs_1

class cv2_cornerHarris_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"blockSize"	: ("INT",),
				"ksize"	: ("INT",),
				"k"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, blockSize, ksize, k, borderType, dst=None):
		ret = apply_function(cv2.cornerHarris, [src, blockSize, ksize, k, dst, borderType], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cornerHarris_0"] = "OpenCV cornerHarris_0"
NODE_CLASS_MAPPINGS	["cornerHarris_0"] = cv2_cornerHarris_0

class cv2_cornerHarris_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"blockSize"	: ("INT",),
				"ksize"	: ("INT",),
				"k"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, blockSize, ksize, k, borderType, dst=None):
		ret = apply_function(cv2.cornerHarris, [src, blockSize, ksize, k, dst, borderType], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cornerHarris_1"] = "OpenCV cornerHarris_1"
NODE_CLASS_MAPPINGS	["cornerHarris_1"] = cv2_cornerHarris_1

class cv2_cornerMinEigenVal_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"blockSize"	: ("INT",),
				"ksize"	: ("INT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, blockSize, ksize, borderType, dst=None):
		ret = apply_function(cv2.cornerMinEigenVal, [src, blockSize, dst, ksize, borderType], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cornerMinEigenVal_0"] = "OpenCV cornerMinEigenVal_0"
NODE_CLASS_MAPPINGS	["cornerMinEigenVal_0"] = cv2_cornerMinEigenVal_0

class cv2_cornerMinEigenVal_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"blockSize"	: ("INT",),
				"ksize"	: ("INT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, blockSize, ksize, borderType, dst=None):
		ret = apply_function(cv2.cornerMinEigenVal, [src, blockSize, dst, ksize, borderType], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cornerMinEigenVal_1"] = "OpenCV cornerMinEigenVal_1"
NODE_CLASS_MAPPINGS	["cornerMinEigenVal_1"] = cv2_cornerMinEigenVal_1

class cv2_cornerSubPix_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"corners"	: ("NPARRAY",),
				"winSize"	: ("STRING",),
				"zeroZone"	: ("STRING",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, corners, winSize, zeroZone, criteria):
		ret = apply_function(cv2.cornerSubPix, [image, corners, winSize, zeroZone, criteria], [0, 1], [2, 3, 4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cornerSubPix_0"] = "OpenCV cornerSubPix_0"
NODE_CLASS_MAPPINGS	["cornerSubPix_0"] = cv2_cornerSubPix_0

class cv2_cornerSubPix_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"corners"	: ("NPARRAY",),
				"winSize"	: ("STRING",),
				"zeroZone"	: ("STRING",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, corners, winSize, zeroZone, criteria):
		ret = apply_function(cv2.cornerSubPix, [image, corners, winSize, zeroZone, criteria], [0, 1], [2, 3, 4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cornerSubPix_1"] = "OpenCV cornerSubPix_1"
NODE_CLASS_MAPPINGS	["cornerSubPix_1"] = cv2_cornerSubPix_1

class cv2_correctMatches_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"F"	: ("NPARRAY",),
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
			},
			'optional': {
				"newPoints1"	: ("NPARRAY",),
				"newPoints2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, F, points1, points2, newPoints1=None, newPoints2=None):
		ret = apply_function(cv2.correctMatches, [F, points1, points2, newPoints1, newPoints2], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["correctMatches_0"] = "OpenCV correctMatches_0"
NODE_CLASS_MAPPINGS	["correctMatches_0"] = cv2_correctMatches_0

class cv2_correctMatches_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"F"	: ("NPARRAY",),
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
			},
			'optional': {
				"newPoints1"	: ("NPARRAY",),
				"newPoints2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, F, points1, points2, newPoints1=None, newPoints2=None):
		ret = apply_function(cv2.correctMatches, [F, points1, points2, newPoints1, newPoints2], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["correctMatches_1"] = "OpenCV correctMatches_1"
NODE_CLASS_MAPPINGS	["correctMatches_1"] = cv2_correctMatches_1

class cv2_countNonZero_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src):
		ret = apply_function(cv2.countNonZero, [src], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["countNonZero_0"] = "OpenCV countNonZero_0"
NODE_CLASS_MAPPINGS	["countNonZero_0"] = cv2_countNonZero_0

class cv2_countNonZero_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src):
		ret = apply_function(cv2.countNonZero, [src], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["countNonZero_1"] = "OpenCV countNonZero_1"
NODE_CLASS_MAPPINGS	["countNonZero_1"] = cv2_countNonZero_1

class cv2_createHanningWindow_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winSize"	: ("STRING",),
				"type"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winSize, type, dst=None):
		ret = apply_function(cv2.createHanningWindow, [winSize, type, dst], [2], [0])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["createHanningWindow_0"] = "OpenCV createHanningWindow_0"
NODE_CLASS_MAPPINGS	["createHanningWindow_0"] = cv2_createHanningWindow_0

class cv2_createHanningWindow_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winSize"	: ("STRING",),
				"type"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winSize, type, dst=None):
		ret = apply_function(cv2.createHanningWindow, [winSize, type, dst], [2], [0])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["createHanningWindow_1"] = "OpenCV createHanningWindow_1"
NODE_CLASS_MAPPINGS	["createHanningWindow_1"] = cv2_createHanningWindow_1

class cv2_cubeRoot_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"val"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, val):
		ret = apply_function(cv2.cubeRoot, [val], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cubeRoot_0"] = "OpenCV cubeRoot_0"
NODE_CLASS_MAPPINGS	["cubeRoot_0"] = cv2_cubeRoot_0

class cv2_currentUIFramework_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("string",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.currentUIFramework, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["currentUIFramework_0"] = "OpenCV currentUIFramework_0"
NODE_CLASS_MAPPINGS	["currentUIFramework_0"] = cv2_currentUIFramework_0

class cv2_cvtColor_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"code"	: ("INT",),
				"dstCn"	: ("INT",),
				"hint"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, code, dstCn, hint, dst=None):
		ret = apply_function(cv2.cvtColor, [src, code, dst, dstCn, hint], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cvtColor_0"] = "OpenCV cvtColor_0"
NODE_CLASS_MAPPINGS	["cvtColor_0"] = cv2_cvtColor_0

class cv2_cvtColor_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"code"	: ("INT",),
				"dstCn"	: ("INT",),
				"hint"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, code, dstCn, hint, dst=None):
		ret = apply_function(cv2.cvtColor, [src, code, dst, dstCn, hint], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cvtColor_1"] = "OpenCV cvtColor_1"
NODE_CLASS_MAPPINGS	["cvtColor_1"] = cv2_cvtColor_1

class cv2_cvtColorTwoPlane_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"code"	: ("INT",),
				"hint"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, code, hint, dst=None):
		ret = apply_function(cv2.cvtColorTwoPlane, [src1, src2, code, dst, hint], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cvtColorTwoPlane_0"] = "OpenCV cvtColorTwoPlane_0"
NODE_CLASS_MAPPINGS	["cvtColorTwoPlane_0"] = cv2_cvtColorTwoPlane_0

class cv2_cvtColorTwoPlane_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"code"	: ("INT",),
				"hint"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, code, hint, dst=None):
		ret = apply_function(cv2.cvtColorTwoPlane, [src1, src2, code, dst, hint], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["cvtColorTwoPlane_1"] = "OpenCV cvtColorTwoPlane_1"
NODE_CLASS_MAPPINGS	["cvtColorTwoPlane_1"] = cv2_cvtColorTwoPlane_1

class cv2_dct_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, dst=None):
		ret = apply_function(cv2.dct, [src, dst, flags], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["dct_0"] = "OpenCV dct_0"
NODE_CLASS_MAPPINGS	["dct_0"] = cv2_dct_0

class cv2_dct_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, dst=None):
		ret = apply_function(cv2.dct, [src, dst, flags], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["dct_1"] = "OpenCV dct_1"
NODE_CLASS_MAPPINGS	["dct_1"] = cv2_dct_1

class cv2_decolor_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"grayscale"	: ("NPARRAY",),
				"color_boost"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, grayscale=None, color_boost=None):
		ret = apply_function(cv2.decolor, [src, grayscale, color_boost], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["decolor_0"] = "OpenCV decolor_0"
NODE_CLASS_MAPPINGS	["decolor_0"] = cv2_decolor_0

class cv2_decolor_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"grayscale"	: ("NPARRAY",),
				"color_boost"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, grayscale=None, color_boost=None):
		ret = apply_function(cv2.decolor, [src, grayscale, color_boost], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["decolor_1"] = "OpenCV decolor_1"
NODE_CLASS_MAPPINGS	["decolor_1"] = cv2_decolor_1

class cv2_decomposeEssentialMat_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"E"	: ("NPARRAY",),
			},
			'optional': {
				"R1"	: ("NPARRAY",),
				"R2"	: ("NPARRAY",),
				"t"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, E, R1=None, R2=None, t=None):
		ret = apply_function(cv2.decomposeEssentialMat, [E, R1, R2, t], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["decomposeEssentialMat_0"] = "OpenCV decomposeEssentialMat_0"
NODE_CLASS_MAPPINGS	["decomposeEssentialMat_0"] = cv2_decomposeEssentialMat_0

class cv2_decomposeEssentialMat_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"E"	: ("NPARRAY",),
			},
			'optional': {
				"R1"	: ("NPARRAY",),
				"R2"	: ("NPARRAY",),
				"t"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, E, R1=None, R2=None, t=None):
		ret = apply_function(cv2.decomposeEssentialMat, [E, R1, R2, t], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["decomposeEssentialMat_1"] = "OpenCV decomposeEssentialMat_1"
NODE_CLASS_MAPPINGS	["decomposeEssentialMat_1"] = cv2_decomposeEssentialMat_1

class cv2_decomposeProjectionMatrix_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"projMatrix"	: ("NPARRAY",),
			},
			'optional': {
				"cameraMatrix"	: ("NPARRAY",),
				"rotMatrix"	: ("NPARRAY",),
				"transVect"	: ("NPARRAY",),
				"rotMatrixX"	: ("NPARRAY",),
				"rotMatrixY"	: ("NPARRAY",),
				"rotMatrixZ"	: ("NPARRAY",),
				"eulerAngles"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2", "nparray_3", "nparray_4", "nparray_5", "nparray_6",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, projMatrix, cameraMatrix=None, rotMatrix=None, transVect=None, rotMatrixX=None, rotMatrixY=None, rotMatrixZ=None, eulerAngles=None):
		ret = apply_function(cv2.decomposeProjectionMatrix, [projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles], [0, 1, 2, 3, 4, 5, 6, 7], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["decomposeProjectionMatrix_0"] = "OpenCV decomposeProjectionMatrix_0"
NODE_CLASS_MAPPINGS	["decomposeProjectionMatrix_0"] = cv2_decomposeProjectionMatrix_0

class cv2_decomposeProjectionMatrix_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"projMatrix"	: ("NPARRAY",),
			},
			'optional': {
				"cameraMatrix"	: ("NPARRAY",),
				"rotMatrix"	: ("NPARRAY",),
				"transVect"	: ("NPARRAY",),
				"rotMatrixX"	: ("NPARRAY",),
				"rotMatrixY"	: ("NPARRAY",),
				"rotMatrixZ"	: ("NPARRAY",),
				"eulerAngles"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2", "nparray_3", "nparray_4", "nparray_5", "nparray_6",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, projMatrix, cameraMatrix=None, rotMatrix=None, transVect=None, rotMatrixX=None, rotMatrixY=None, rotMatrixZ=None, eulerAngles=None):
		ret = apply_function(cv2.decomposeProjectionMatrix, [projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles], [0, 1, 2, 3, 4, 5, 6, 7], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["decomposeProjectionMatrix_1"] = "OpenCV decomposeProjectionMatrix_1"
NODE_CLASS_MAPPINGS	["decomposeProjectionMatrix_1"] = cv2_decomposeProjectionMatrix_1

class cv2_demosaicing_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"code"	: ("INT",),
				"dstCn"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, code, dstCn, dst=None):
		ret = apply_function(cv2.demosaicing, [src, code, dst, dstCn], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["demosaicing_0"] = "OpenCV demosaicing_0"
NODE_CLASS_MAPPINGS	["demosaicing_0"] = cv2_demosaicing_0

class cv2_demosaicing_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"code"	: ("INT",),
				"dstCn"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, code, dstCn, dst=None):
		ret = apply_function(cv2.demosaicing, [src, code, dst, dstCn], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["demosaicing_1"] = "OpenCV demosaicing_1"
NODE_CLASS_MAPPINGS	["demosaicing_1"] = cv2_demosaicing_1

class cv2_destroyAllWindows_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.destroyAllWindows, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["destroyAllWindows_0"] = "OpenCV destroyAllWindows_0"
NODE_CLASS_MAPPINGS	["destroyAllWindows_0"] = cv2_destroyAllWindows_0

class cv2_destroyWindow_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname):
		ret = apply_function(cv2.destroyWindow, [winname], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["destroyWindow_0"] = "OpenCV destroyWindow_0"
NODE_CLASS_MAPPINGS	["destroyWindow_0"] = cv2_destroyWindow_0

class cv2_detailEnhance_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sigma_s"	: ("FLOAT",),
				"sigma_r"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sigma_s, sigma_r, dst=None):
		ret = apply_function(cv2.detailEnhance, [src, dst, sigma_s, sigma_r], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["detailEnhance_0"] = "OpenCV detailEnhance_0"
NODE_CLASS_MAPPINGS	["detailEnhance_0"] = cv2_detailEnhance_0

class cv2_detailEnhance_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sigma_s"	: ("FLOAT",),
				"sigma_r"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sigma_s, sigma_r, dst=None):
		ret = apply_function(cv2.detailEnhance, [src, dst, sigma_s, sigma_r], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["detailEnhance_1"] = "OpenCV detailEnhance_1"
NODE_CLASS_MAPPINGS	["detailEnhance_1"] = cv2_detailEnhance_1

class cv2_determinant_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"mtx"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, mtx):
		ret = apply_function(cv2.determinant, [mtx], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["determinant_0"] = "OpenCV determinant_0"
NODE_CLASS_MAPPINGS	["determinant_0"] = cv2_determinant_0

class cv2_determinant_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"mtx"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, mtx):
		ret = apply_function(cv2.determinant, [mtx], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["determinant_1"] = "OpenCV determinant_1"
NODE_CLASS_MAPPINGS	["determinant_1"] = cv2_determinant_1

class cv2_dft_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"nonzeroRows"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, nonzeroRows, dst=None):
		ret = apply_function(cv2.dft, [src, dst, flags, nonzeroRows], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["dft_0"] = "OpenCV dft_0"
NODE_CLASS_MAPPINGS	["dft_0"] = cv2_dft_0

class cv2_dft_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"nonzeroRows"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, nonzeroRows, dst=None):
		ret = apply_function(cv2.dft, [src, dst, flags, nonzeroRows], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["dft_1"] = "OpenCV dft_1"
NODE_CLASS_MAPPINGS	["dft_1"] = cv2_dft_1

class cv2_dilate_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"kernel"	: ("NPARRAY",),
				"anchor"	: ("STRING",),
				"iterations"	: ("INT",),
				"borderType"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, kernel, anchor, iterations, borderType, borderValue, dst=None):
		ret = apply_function(cv2.dilate, [src, kernel, dst, anchor, iterations, borderType, borderValue], [0, 1, 2], [3, 6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["dilate_0"] = "OpenCV dilate_0"
NODE_CLASS_MAPPINGS	["dilate_0"] = cv2_dilate_0

class cv2_dilate_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"kernel"	: ("NPARRAY",),
				"anchor"	: ("STRING",),
				"iterations"	: ("INT",),
				"borderType"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, kernel, anchor, iterations, borderType, borderValue, dst=None):
		ret = apply_function(cv2.dilate, [src, kernel, dst, anchor, iterations, borderType, borderValue], [0, 1, 2], [3, 6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["dilate_1"] = "OpenCV dilate_1"
NODE_CLASS_MAPPINGS	["dilate_1"] = cv2_dilate_1

class cv2_displayOverlay_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"text"	: ("STRING",),
				"delayms"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, text, delayms):
		ret = apply_function(cv2.displayOverlay, [winname, text, delayms], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["displayOverlay_0"] = "OpenCV displayOverlay_0"
NODE_CLASS_MAPPINGS	["displayOverlay_0"] = cv2_displayOverlay_0

class cv2_displayStatusBar_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"text"	: ("STRING",),
				"delayms"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, text, delayms):
		ret = apply_function(cv2.displayStatusBar, [winname, text, delayms], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["displayStatusBar_0"] = "OpenCV displayStatusBar_0"
NODE_CLASS_MAPPINGS	["displayStatusBar_0"] = cv2_displayStatusBar_0

class cv2_distanceTransform_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"distanceType"	: ("INT",),
				"maskSize"	: ("INT",),
				"dstType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, distanceType, maskSize, dstType, dst=None):
		ret = apply_function(cv2.distanceTransform, [src, distanceType, maskSize, dst, dstType], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["distanceTransform_0"] = "OpenCV distanceTransform_0"
NODE_CLASS_MAPPINGS	["distanceTransform_0"] = cv2_distanceTransform_0

class cv2_distanceTransform_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"distanceType"	: ("INT",),
				"maskSize"	: ("INT",),
				"dstType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, distanceType, maskSize, dstType, dst=None):
		ret = apply_function(cv2.distanceTransform, [src, distanceType, maskSize, dst, dstType], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["distanceTransform_1"] = "OpenCV distanceTransform_1"
NODE_CLASS_MAPPINGS	["distanceTransform_1"] = cv2_distanceTransform_1

class cv2_distanceTransformWithLabels_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"distanceType"	: ("INT",),
				"maskSize"	: ("INT",),
				"labelType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"labels"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, distanceType, maskSize, labelType, dst=None, labels=None):
		ret = apply_function(cv2.distanceTransformWithLabels, [src, distanceType, maskSize, dst, labels, labelType], [0, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["distanceTransformWithLabels_0"] = "OpenCV distanceTransformWithLabels_0"
NODE_CLASS_MAPPINGS	["distanceTransformWithLabels_0"] = cv2_distanceTransformWithLabels_0

class cv2_distanceTransformWithLabels_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"distanceType"	: ("INT",),
				"maskSize"	: ("INT",),
				"labelType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"labels"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, distanceType, maskSize, labelType, dst=None, labels=None):
		ret = apply_function(cv2.distanceTransformWithLabels, [src, distanceType, maskSize, dst, labels, labelType], [0, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["distanceTransformWithLabels_1"] = "OpenCV distanceTransformWithLabels_1"
NODE_CLASS_MAPPINGS	["distanceTransformWithLabels_1"] = cv2_distanceTransformWithLabels_1

class cv2_divSpectrums_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"a"	: ("NPARRAY",),
				"b"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"conjB"	: ("BOOLEAN",),
			},
			'optional': {
				"c"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, a, b, flags, conjB, c=None):
		ret = apply_function(cv2.divSpectrums, [a, b, flags, c, conjB], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["divSpectrums_0"] = "OpenCV divSpectrums_0"
NODE_CLASS_MAPPINGS	["divSpectrums_0"] = cv2_divSpectrums_0

class cv2_divSpectrums_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"a"	: ("NPARRAY",),
				"b"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"conjB"	: ("BOOLEAN",),
			},
			'optional': {
				"c"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, a, b, flags, conjB, c=None):
		ret = apply_function(cv2.divSpectrums, [a, b, flags, c, conjB], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["divSpectrums_1"] = "OpenCV divSpectrums_1"
NODE_CLASS_MAPPINGS	["divSpectrums_1"] = cv2_divSpectrums_1

class cv2_divide_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"scale"	: ("FLOAT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, scale, dtype, dst=None):
		ret = apply_function(cv2.divide, [src1, src2, dst, scale, dtype], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["divide_0"] = "OpenCV divide_0"
NODE_CLASS_MAPPINGS	["divide_0"] = cv2_divide_0

class cv2_divide_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"scale"	: ("FLOAT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, scale, dtype, dst=None):
		ret = apply_function(cv2.divide, [src1, src2, dst, scale, dtype], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["divide_1"] = "OpenCV divide_1"
NODE_CLASS_MAPPINGS	["divide_1"] = cv2_divide_1

class cv2_divide_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"scale"	: ("FLOAT",),
				"src2"	: ("NPARRAY",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, scale, src2, dtype, dst=None):
		ret = apply_function(cv2.divide, [scale, src2, dst, dtype], [1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["divide_2"] = "OpenCV divide_2"
NODE_CLASS_MAPPINGS	["divide_2"] = cv2_divide_2

class cv2_divide_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"scale"	: ("FLOAT",),
				"src2"	: ("NPARRAY",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, scale, src2, dtype, dst=None):
		ret = apply_function(cv2.divide, [scale, src2, dst, dtype], [1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["divide_3"] = "OpenCV divide_3"
NODE_CLASS_MAPPINGS	["divide_3"] = cv2_divide_3

class cv2_drawChessboardCorners_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patternSize"	: ("STRING",),
				"corners"	: ("NPARRAY",),
				"patternWasFound"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patternSize, corners, patternWasFound):
		ret = apply_function(cv2.drawChessboardCorners, [image, patternSize, corners, patternWasFound], [0, 2], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["drawChessboardCorners_0"] = "OpenCV drawChessboardCorners_0"
NODE_CLASS_MAPPINGS	["drawChessboardCorners_0"] = cv2_drawChessboardCorners_0

class cv2_drawChessboardCorners_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patternSize"	: ("STRING",),
				"corners"	: ("NPARRAY",),
				"patternWasFound"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patternSize, corners, patternWasFound):
		ret = apply_function(cv2.drawChessboardCorners, [image, patternSize, corners, patternWasFound], [0, 2], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["drawChessboardCorners_1"] = "OpenCV drawChessboardCorners_1"
NODE_CLASS_MAPPINGS	["drawChessboardCorners_1"] = cv2_drawChessboardCorners_1

class cv2_drawFrameAxes_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
				"length"	: ("FLOAT",),
				"thickness"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, cameraMatrix, distCoeffs, rvec, tvec, length, thickness):
		ret = apply_function(cv2.drawFrameAxes, [image, cameraMatrix, distCoeffs, rvec, tvec, length, thickness], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["drawFrameAxes_0"] = "OpenCV drawFrameAxes_0"
NODE_CLASS_MAPPINGS	["drawFrameAxes_0"] = cv2_drawFrameAxes_0

class cv2_drawFrameAxes_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
				"length"	: ("FLOAT",),
				"thickness"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, cameraMatrix, distCoeffs, rvec, tvec, length, thickness):
		ret = apply_function(cv2.drawFrameAxes, [image, cameraMatrix, distCoeffs, rvec, tvec, length, thickness], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["drawFrameAxes_1"] = "OpenCV drawFrameAxes_1"
NODE_CLASS_MAPPINGS	["drawFrameAxes_1"] = cv2_drawFrameAxes_1

class cv2_drawMarker_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"position"	: ("STRING",),
				"color"	: ("STRING",),
				"markerType"	: ("INT",),
				"markerSize"	: ("INT",),
				"thickness"	: ("INT",),
				"line_type"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, position, color, markerType, markerSize, thickness, line_type):
		ret = apply_function(cv2.drawMarker, [img, position, color, markerType, markerSize, thickness, line_type], [0], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["drawMarker_0"] = "OpenCV drawMarker_0"
NODE_CLASS_MAPPINGS	["drawMarker_0"] = cv2_drawMarker_0

class cv2_drawMarker_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"position"	: ("STRING",),
				"color"	: ("STRING",),
				"markerType"	: ("INT",),
				"markerSize"	: ("INT",),
				"thickness"	: ("INT",),
				"line_type"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, position, color, markerType, markerSize, thickness, line_type):
		ret = apply_function(cv2.drawMarker, [img, position, color, markerType, markerSize, thickness, line_type], [0], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["drawMarker_1"] = "OpenCV drawMarker_1"
NODE_CLASS_MAPPINGS	["drawMarker_1"] = cv2_drawMarker_1

class cv2_edgePreservingFilter_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"sigma_s"	: ("FLOAT",),
				"sigma_r"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, sigma_s, sigma_r, dst=None):
		ret = apply_function(cv2.edgePreservingFilter, [src, dst, flags, sigma_s, sigma_r], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["edgePreservingFilter_0"] = "OpenCV edgePreservingFilter_0"
NODE_CLASS_MAPPINGS	["edgePreservingFilter_0"] = cv2_edgePreservingFilter_0

class cv2_edgePreservingFilter_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"sigma_s"	: ("FLOAT",),
				"sigma_r"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, sigma_s, sigma_r, dst=None):
		ret = apply_function(cv2.edgePreservingFilter, [src, dst, flags, sigma_s, sigma_r], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["edgePreservingFilter_1"] = "OpenCV edgePreservingFilter_1"
NODE_CLASS_MAPPINGS	["edgePreservingFilter_1"] = cv2_edgePreservingFilter_1

class cv2_eigen_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"eigenvalues"	: ("NPARRAY",),
				"eigenvectors"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, eigenvalues=None, eigenvectors=None):
		ret = apply_function(cv2.eigen, [src, eigenvalues, eigenvectors], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["eigen_0"] = "OpenCV eigen_0"
NODE_CLASS_MAPPINGS	["eigen_0"] = cv2_eigen_0

class cv2_eigen_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"eigenvalues"	: ("NPARRAY",),
				"eigenvectors"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, eigenvalues=None, eigenvectors=None):
		ret = apply_function(cv2.eigen, [src, eigenvalues, eigenvectors], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["eigen_1"] = "OpenCV eigen_1"
NODE_CLASS_MAPPINGS	["eigen_1"] = cv2_eigen_1

class cv2_eigenNonSymmetric_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"eigenvalues"	: ("NPARRAY",),
				"eigenvectors"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, eigenvalues=None, eigenvectors=None):
		ret = apply_function(cv2.eigenNonSymmetric, [src, eigenvalues, eigenvectors], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["eigenNonSymmetric_0"] = "OpenCV eigenNonSymmetric_0"
NODE_CLASS_MAPPINGS	["eigenNonSymmetric_0"] = cv2_eigenNonSymmetric_0

class cv2_eigenNonSymmetric_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"eigenvalues"	: ("NPARRAY",),
				"eigenvectors"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, eigenvalues=None, eigenvectors=None):
		ret = apply_function(cv2.eigenNonSymmetric, [src, eigenvalues, eigenvectors], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["eigenNonSymmetric_1"] = "OpenCV eigenNonSymmetric_1"
NODE_CLASS_MAPPINGS	["eigenNonSymmetric_1"] = cv2_eigenNonSymmetric_1

class cv2_ellipse_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"center"	: ("STRING",),
				"axes"	: ("STRING",),
				"angle"	: ("FLOAT",),
				"startAngle"	: ("FLOAT",),
				"endAngle"	: ("FLOAT",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift):
		ret = apply_function(cv2.ellipse, [img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift], [0], [1, 2, 6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["ellipse_0"] = "OpenCV ellipse_0"
NODE_CLASS_MAPPINGS	["ellipse_0"] = cv2_ellipse_0

class cv2_ellipse_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"center"	: ("STRING",),
				"axes"	: ("STRING",),
				"angle"	: ("FLOAT",),
				"startAngle"	: ("FLOAT",),
				"endAngle"	: ("FLOAT",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift):
		ret = apply_function(cv2.ellipse, [img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift], [0], [1, 2, 6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["ellipse_1"] = "OpenCV ellipse_1"
NODE_CLASS_MAPPINGS	["ellipse_1"] = cv2_ellipse_1

class cv2_ellipse_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"box"	: ("STRING",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, box, color, thickness, lineType):
		ret = apply_function(cv2.ellipse, [img, box, color, thickness, lineType], [0], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["ellipse_2"] = "OpenCV ellipse_2"
NODE_CLASS_MAPPINGS	["ellipse_2"] = cv2_ellipse_2

class cv2_ellipse_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"box"	: ("STRING",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, box, color, thickness, lineType):
		ret = apply_function(cv2.ellipse, [img, box, color, thickness, lineType], [0], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["ellipse_3"] = "OpenCV ellipse_3"
NODE_CLASS_MAPPINGS	["ellipse_3"] = cv2_ellipse_3

class cv2_equalizeHist_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.equalizeHist, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["equalizeHist_0"] = "OpenCV equalizeHist_0"
NODE_CLASS_MAPPINGS	["equalizeHist_0"] = cv2_equalizeHist_0

class cv2_equalizeHist_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.equalizeHist, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["equalizeHist_1"] = "OpenCV equalizeHist_1"
NODE_CLASS_MAPPINGS	["equalizeHist_1"] = cv2_equalizeHist_1

class cv2_erode_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"kernel"	: ("NPARRAY",),
				"anchor"	: ("STRING",),
				"iterations"	: ("INT",),
				"borderType"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, kernel, anchor, iterations, borderType, borderValue, dst=None):
		ret = apply_function(cv2.erode, [src, kernel, dst, anchor, iterations, borderType, borderValue], [0, 1, 2], [3, 6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["erode_0"] = "OpenCV erode_0"
NODE_CLASS_MAPPINGS	["erode_0"] = cv2_erode_0

class cv2_erode_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"kernel"	: ("NPARRAY",),
				"anchor"	: ("STRING",),
				"iterations"	: ("INT",),
				"borderType"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, kernel, anchor, iterations, borderType, borderValue, dst=None):
		ret = apply_function(cv2.erode, [src, kernel, dst, anchor, iterations, borderType, borderValue], [0, 1, 2], [3, 6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["erode_1"] = "OpenCV erode_1"
NODE_CLASS_MAPPINGS	["erode_1"] = cv2_erode_1

class cv2_estimateAffine2D_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"from_"	: ("NPARRAY",),
				"to"	: ("NPARRAY",),
				"method"	: ("INT",),
				"ransacReprojThreshold"	: ("FLOAT",),
				"maxIters"	: ("INT",),
				"confidence"	: ("FLOAT",),
				"refineIters"	: ("INT",),
			},
			'optional': {
				"inliers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, from_, to, method, ransacReprojThreshold, maxIters, confidence, refineIters, inliers=None):
		ret = apply_function(cv2.estimateAffine2D, [from_, to, inliers, method, ransacReprojThreshold, maxIters, confidence, refineIters], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateAffine2D_0"] = "OpenCV estimateAffine2D_0"
NODE_CLASS_MAPPINGS	["estimateAffine2D_0"] = cv2_estimateAffine2D_0

class cv2_estimateAffine2D_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"from_"	: ("NPARRAY",),
				"to"	: ("NPARRAY",),
				"method"	: ("INT",),
				"ransacReprojThreshold"	: ("FLOAT",),
				"maxIters"	: ("INT",),
				"confidence"	: ("FLOAT",),
				"refineIters"	: ("INT",),
			},
			'optional': {
				"inliers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, from_, to, method, ransacReprojThreshold, maxIters, confidence, refineIters, inliers=None):
		ret = apply_function(cv2.estimateAffine2D, [from_, to, inliers, method, ransacReprojThreshold, maxIters, confidence, refineIters], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateAffine2D_1"] = "OpenCV estimateAffine2D_1"
NODE_CLASS_MAPPINGS	["estimateAffine2D_1"] = cv2_estimateAffine2D_1

class cv2_estimateAffine3D_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"ransacThreshold"	: ("FLOAT",),
				"confidence"	: ("FLOAT",),
			},
			'optional': {
				"out"	: ("NPARRAY",),
				"inliers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, ransacThreshold, confidence, out=None, inliers=None):
		ret = apply_function(cv2.estimateAffine3D, [src, dst, out, inliers, ransacThreshold, confidence], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateAffine3D_0"] = "OpenCV estimateAffine3D_0"
NODE_CLASS_MAPPINGS	["estimateAffine3D_0"] = cv2_estimateAffine3D_0

class cv2_estimateAffine3D_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"ransacThreshold"	: ("FLOAT",),
				"confidence"	: ("FLOAT",),
			},
			'optional': {
				"out"	: ("NPARRAY",),
				"inliers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, ransacThreshold, confidence, out=None, inliers=None):
		ret = apply_function(cv2.estimateAffine3D, [src, dst, out, inliers, ransacThreshold, confidence], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateAffine3D_1"] = "OpenCV estimateAffine3D_1"
NODE_CLASS_MAPPINGS	["estimateAffine3D_1"] = cv2_estimateAffine3D_1

class cv2_estimateAffine3D_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"force_rotation"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY", "FLOAT",)
	RETURN_NAMES	= ("nparray", "float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, force_rotation):
		ret = apply_function(cv2.estimateAffine3D, [src, dst, force_rotation], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateAffine3D_2"] = "OpenCV estimateAffine3D_2"
NODE_CLASS_MAPPINGS	["estimateAffine3D_2"] = cv2_estimateAffine3D_2

class cv2_estimateAffine3D_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"force_rotation"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY", "FLOAT",)
	RETURN_NAMES	= ("nparray", "float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, force_rotation):
		ret = apply_function(cv2.estimateAffine3D, [src, dst, force_rotation], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateAffine3D_3"] = "OpenCV estimateAffine3D_3"
NODE_CLASS_MAPPINGS	["estimateAffine3D_3"] = cv2_estimateAffine3D_3

class cv2_estimateAffinePartial2D_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"from_"	: ("NPARRAY",),
				"to"	: ("NPARRAY",),
				"method"	: ("INT",),
				"ransacReprojThreshold"	: ("FLOAT",),
				"maxIters"	: ("INT",),
				"confidence"	: ("FLOAT",),
				"refineIters"	: ("INT",),
			},
			'optional': {
				"inliers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, from_, to, method, ransacReprojThreshold, maxIters, confidence, refineIters, inliers=None):
		ret = apply_function(cv2.estimateAffinePartial2D, [from_, to, inliers, method, ransacReprojThreshold, maxIters, confidence, refineIters], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateAffinePartial2D_0"] = "OpenCV estimateAffinePartial2D_0"
NODE_CLASS_MAPPINGS	["estimateAffinePartial2D_0"] = cv2_estimateAffinePartial2D_0

class cv2_estimateAffinePartial2D_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"from_"	: ("NPARRAY",),
				"to"	: ("NPARRAY",),
				"method"	: ("INT",),
				"ransacReprojThreshold"	: ("FLOAT",),
				"maxIters"	: ("INT",),
				"confidence"	: ("FLOAT",),
				"refineIters"	: ("INT",),
			},
			'optional': {
				"inliers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, from_, to, method, ransacReprojThreshold, maxIters, confidence, refineIters, inliers=None):
		ret = apply_function(cv2.estimateAffinePartial2D, [from_, to, inliers, method, ransacReprojThreshold, maxIters, confidence, refineIters], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateAffinePartial2D_1"] = "OpenCV estimateAffinePartial2D_1"
NODE_CLASS_MAPPINGS	["estimateAffinePartial2D_1"] = cv2_estimateAffinePartial2D_1

class cv2_estimateChessboardSharpness_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patternSize"	: ("STRING",),
				"corners"	: ("NPARRAY",),
				"rise_distance"	: ("FLOAT",),
				"vertical"	: ("BOOLEAN",),
			},
			'optional': {
				"sharpness"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("STRING", "NPARRAY",)
	RETURN_NAMES	= ("literal", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patternSize, corners, rise_distance, vertical, sharpness=None):
		ret = apply_function(cv2.estimateChessboardSharpness, [image, patternSize, corners, rise_distance, vertical, sharpness], [0, 2, 5], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateChessboardSharpness_0"] = "OpenCV estimateChessboardSharpness_0"
NODE_CLASS_MAPPINGS	["estimateChessboardSharpness_0"] = cv2_estimateChessboardSharpness_0

class cv2_estimateChessboardSharpness_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patternSize"	: ("STRING",),
				"corners"	: ("NPARRAY",),
				"rise_distance"	: ("FLOAT",),
				"vertical"	: ("BOOLEAN",),
			},
			'optional': {
				"sharpness"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("STRING", "NPARRAY",)
	RETURN_NAMES	= ("literal", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patternSize, corners, rise_distance, vertical, sharpness=None):
		ret = apply_function(cv2.estimateChessboardSharpness, [image, patternSize, corners, rise_distance, vertical, sharpness], [0, 2, 5], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateChessboardSharpness_1"] = "OpenCV estimateChessboardSharpness_1"
NODE_CLASS_MAPPINGS	["estimateChessboardSharpness_1"] = cv2_estimateChessboardSharpness_1

class cv2_estimateTranslation3D_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"ransacThreshold"	: ("FLOAT",),
				"confidence"	: ("FLOAT",),
			},
			'optional': {
				"out"	: ("NPARRAY",),
				"inliers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, ransacThreshold, confidence, out=None, inliers=None):
		ret = apply_function(cv2.estimateTranslation3D, [src, dst, out, inliers, ransacThreshold, confidence], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateTranslation3D_0"] = "OpenCV estimateTranslation3D_0"
NODE_CLASS_MAPPINGS	["estimateTranslation3D_0"] = cv2_estimateTranslation3D_0

class cv2_estimateTranslation3D_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"ransacThreshold"	: ("FLOAT",),
				"confidence"	: ("FLOAT",),
			},
			'optional': {
				"out"	: ("NPARRAY",),
				"inliers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, ransacThreshold, confidence, out=None, inliers=None):
		ret = apply_function(cv2.estimateTranslation3D, [src, dst, out, inliers, ransacThreshold, confidence], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["estimateTranslation3D_1"] = "OpenCV estimateTranslation3D_1"
NODE_CLASS_MAPPINGS	["estimateTranslation3D_1"] = cv2_estimateTranslation3D_1

class cv2_exp_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.exp, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["exp_0"] = "OpenCV exp_0"
NODE_CLASS_MAPPINGS	["exp_0"] = cv2_exp_0

class cv2_exp_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.exp, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["exp_1"] = "OpenCV exp_1"
NODE_CLASS_MAPPINGS	["exp_1"] = cv2_exp_1

class cv2_extractChannel_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"coi"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, coi, dst=None):
		ret = apply_function(cv2.extractChannel, [src, coi, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["extractChannel_0"] = "OpenCV extractChannel_0"
NODE_CLASS_MAPPINGS	["extractChannel_0"] = cv2_extractChannel_0

class cv2_extractChannel_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"coi"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, coi, dst=None):
		ret = apply_function(cv2.extractChannel, [src, coi, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["extractChannel_1"] = "OpenCV extractChannel_1"
NODE_CLASS_MAPPINGS	["extractChannel_1"] = cv2_extractChannel_1

class cv2_fastAtan2_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"y"	: ("FLOAT",),
				"x"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, y, x):
		ret = apply_function(cv2.fastAtan2, [y, x], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fastAtan2_0"] = "OpenCV fastAtan2_0"
NODE_CLASS_MAPPINGS	["fastAtan2_0"] = cv2_fastAtan2_0

class cv2_fastNlMeansDenoising_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"h"	: ("FLOAT",),
				"templateWindowSize"	: ("INT",),
				"searchWindowSize"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, h, templateWindowSize, searchWindowSize, dst=None):
		ret = apply_function(cv2.fastNlMeansDenoising, [src, dst, h, templateWindowSize, searchWindowSize], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fastNlMeansDenoising_0"] = "OpenCV fastNlMeansDenoising_0"
NODE_CLASS_MAPPINGS	["fastNlMeansDenoising_0"] = cv2_fastNlMeansDenoising_0

class cv2_fastNlMeansDenoising_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"h"	: ("FLOAT",),
				"templateWindowSize"	: ("INT",),
				"searchWindowSize"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, h, templateWindowSize, searchWindowSize, dst=None):
		ret = apply_function(cv2.fastNlMeansDenoising, [src, dst, h, templateWindowSize, searchWindowSize], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fastNlMeansDenoising_1"] = "OpenCV fastNlMeansDenoising_1"
NODE_CLASS_MAPPINGS	["fastNlMeansDenoising_1"] = cv2_fastNlMeansDenoising_1

class cv2_fastNlMeansDenoisingColored_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"h"	: ("FLOAT",),
				"hColor"	: ("FLOAT",),
				"templateWindowSize"	: ("INT",),
				"searchWindowSize"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, h, hColor, templateWindowSize, searchWindowSize, dst=None):
		ret = apply_function(cv2.fastNlMeansDenoisingColored, [src, dst, h, hColor, templateWindowSize, searchWindowSize], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fastNlMeansDenoisingColored_0"] = "OpenCV fastNlMeansDenoisingColored_0"
NODE_CLASS_MAPPINGS	["fastNlMeansDenoisingColored_0"] = cv2_fastNlMeansDenoisingColored_0

class cv2_fastNlMeansDenoisingColored_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"h"	: ("FLOAT",),
				"hColor"	: ("FLOAT",),
				"templateWindowSize"	: ("INT",),
				"searchWindowSize"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, h, hColor, templateWindowSize, searchWindowSize, dst=None):
		ret = apply_function(cv2.fastNlMeansDenoisingColored, [src, dst, h, hColor, templateWindowSize, searchWindowSize], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fastNlMeansDenoisingColored_1"] = "OpenCV fastNlMeansDenoisingColored_1"
NODE_CLASS_MAPPINGS	["fastNlMeansDenoisingColored_1"] = cv2_fastNlMeansDenoisingColored_1

class cv2_fillConvexPoly_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"points"	: ("NPARRAY",),
				"color"	: ("STRING",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, points, color, lineType, shift):
		ret = apply_function(cv2.fillConvexPoly, [img, points, color, lineType, shift], [0, 1], [2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fillConvexPoly_0"] = "OpenCV fillConvexPoly_0"
NODE_CLASS_MAPPINGS	["fillConvexPoly_0"] = cv2_fillConvexPoly_0

class cv2_fillConvexPoly_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"points"	: ("NPARRAY",),
				"color"	: ("STRING",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, points, color, lineType, shift):
		ret = apply_function(cv2.fillConvexPoly, [img, points, color, lineType, shift], [0, 1], [2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fillConvexPoly_1"] = "OpenCV fillConvexPoly_1"
NODE_CLASS_MAPPINGS	["fillConvexPoly_1"] = cv2_fillConvexPoly_1

class cv2_filter2D_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"kernel"	: ("NPARRAY",),
				"anchor"	: ("STRING",),
				"delta"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, kernel, anchor, delta, borderType, dst=None):
		ret = apply_function(cv2.filter2D, [src, ddepth, kernel, dst, anchor, delta, borderType], [0, 2, 3], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["filter2D_0"] = "OpenCV filter2D_0"
NODE_CLASS_MAPPINGS	["filter2D_0"] = cv2_filter2D_0

class cv2_filter2D_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"kernel"	: ("NPARRAY",),
				"anchor"	: ("STRING",),
				"delta"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, kernel, anchor, delta, borderType, dst=None):
		ret = apply_function(cv2.filter2D, [src, ddepth, kernel, dst, anchor, delta, borderType], [0, 2, 3], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["filter2D_1"] = "OpenCV filter2D_1"
NODE_CLASS_MAPPINGS	["filter2D_1"] = cv2_filter2D_1

class cv2_filterSpeckles_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"newVal"	: ("FLOAT",),
				"maxSpeckleSize"	: ("INT",),
				"maxDiff"	: ("FLOAT",),
			},
			'optional': {
				"buf"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, newVal, maxSpeckleSize, maxDiff, buf=None):
		ret = apply_function(cv2.filterSpeckles, [img, newVal, maxSpeckleSize, maxDiff, buf], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["filterSpeckles_0"] = "OpenCV filterSpeckles_0"
NODE_CLASS_MAPPINGS	["filterSpeckles_0"] = cv2_filterSpeckles_0

class cv2_filterSpeckles_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"newVal"	: ("FLOAT",),
				"maxSpeckleSize"	: ("INT",),
				"maxDiff"	: ("FLOAT",),
			},
			'optional': {
				"buf"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, newVal, maxSpeckleSize, maxDiff, buf=None):
		ret = apply_function(cv2.filterSpeckles, [img, newVal, maxSpeckleSize, maxDiff, buf], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["filterSpeckles_1"] = "OpenCV filterSpeckles_1"
NODE_CLASS_MAPPINGS	["filterSpeckles_1"] = cv2_filterSpeckles_1

class cv2_find4QuadCornerSubpix_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"corners"	: ("NPARRAY",),
				"region_size"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, corners, region_size):
		ret = apply_function(cv2.find4QuadCornerSubpix, [img, corners, region_size], [0, 1], [2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["find4QuadCornerSubpix_0"] = "OpenCV find4QuadCornerSubpix_0"
NODE_CLASS_MAPPINGS	["find4QuadCornerSubpix_0"] = cv2_find4QuadCornerSubpix_0

class cv2_find4QuadCornerSubpix_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"corners"	: ("NPARRAY",),
				"region_size"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, corners, region_size):
		ret = apply_function(cv2.find4QuadCornerSubpix, [img, corners, region_size], [0, 1], [2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["find4QuadCornerSubpix_1"] = "OpenCV find4QuadCornerSubpix_1"
NODE_CLASS_MAPPINGS	["find4QuadCornerSubpix_1"] = cv2_find4QuadCornerSubpix_1

class cv2_findChessboardCorners_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patternSize"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patternSize, flags, corners=None):
		ret = apply_function(cv2.findChessboardCorners, [image, patternSize, corners, flags], [0, 2], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findChessboardCorners_0"] = "OpenCV findChessboardCorners_0"
NODE_CLASS_MAPPINGS	["findChessboardCorners_0"] = cv2_findChessboardCorners_0

class cv2_findChessboardCorners_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patternSize"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patternSize, flags, corners=None):
		ret = apply_function(cv2.findChessboardCorners, [image, patternSize, corners, flags], [0, 2], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findChessboardCorners_1"] = "OpenCV findChessboardCorners_1"
NODE_CLASS_MAPPINGS	["findChessboardCorners_1"] = cv2_findChessboardCorners_1

class cv2_findChessboardCornersSB_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patternSize"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patternSize, flags, corners=None):
		ret = apply_function(cv2.findChessboardCornersSB, [image, patternSize, corners, flags], [0, 2], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findChessboardCornersSB_0"] = "OpenCV findChessboardCornersSB_0"
NODE_CLASS_MAPPINGS	["findChessboardCornersSB_0"] = cv2_findChessboardCornersSB_0

class cv2_findChessboardCornersSB_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patternSize"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patternSize, flags, corners=None):
		ret = apply_function(cv2.findChessboardCornersSB, [image, patternSize, corners, flags], [0, 2], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findChessboardCornersSB_1"] = "OpenCV findChessboardCornersSB_1"
NODE_CLASS_MAPPINGS	["findChessboardCornersSB_1"] = cv2_findChessboardCornersSB_1

class cv2_findChessboardCornersSBWithMeta_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patternSize"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
				"meta"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patternSize, flags, corners=None, meta=None):
		ret = apply_function(cv2.findChessboardCornersSBWithMeta, [image, patternSize, flags, corners, meta], [0, 3, 4], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findChessboardCornersSBWithMeta_0"] = "OpenCV findChessboardCornersSBWithMeta_0"
NODE_CLASS_MAPPINGS	["findChessboardCornersSBWithMeta_0"] = cv2_findChessboardCornersSBWithMeta_0

class cv2_findChessboardCornersSBWithMeta_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patternSize"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
				"meta"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patternSize, flags, corners=None, meta=None):
		ret = apply_function(cv2.findChessboardCornersSBWithMeta, [image, patternSize, flags, corners, meta], [0, 3, 4], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findChessboardCornersSBWithMeta_1"] = "OpenCV findChessboardCornersSBWithMeta_1"
NODE_CLASS_MAPPINGS	["findChessboardCornersSBWithMeta_1"] = cv2_findChessboardCornersSBWithMeta_1

class cv2_findEssentialMat_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"method"	: ("INT",),
				"prob"	: ("FLOAT",),
				"threshold"	: ("FLOAT",),
				"maxIters"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, cameraMatrix, method, prob, threshold, maxIters, mask=None):
		ret = apply_function(cv2.findEssentialMat, [points1, points2, cameraMatrix, method, prob, threshold, maxIters, mask], [0, 1, 2, 7], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findEssentialMat_0"] = "OpenCV findEssentialMat_0"
NODE_CLASS_MAPPINGS	["findEssentialMat_0"] = cv2_findEssentialMat_0

class cv2_findEssentialMat_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"method"	: ("INT",),
				"prob"	: ("FLOAT",),
				"threshold"	: ("FLOAT",),
				"maxIters"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, cameraMatrix, method, prob, threshold, maxIters, mask=None):
		ret = apply_function(cv2.findEssentialMat, [points1, points2, cameraMatrix, method, prob, threshold, maxIters, mask], [0, 1, 2, 7], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findEssentialMat_1"] = "OpenCV findEssentialMat_1"
NODE_CLASS_MAPPINGS	["findEssentialMat_1"] = cv2_findEssentialMat_1

class cv2_findEssentialMat_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"focal"	: ("FLOAT",),
				"pp"	: ("STRING",),
				"method"	: ("INT",),
				"prob"	: ("FLOAT",),
				"threshold"	: ("FLOAT",),
				"maxIters"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, focal, pp, method, prob, threshold, maxIters, mask=None):
		ret = apply_function(cv2.findEssentialMat, [points1, points2, focal, pp, method, prob, threshold, maxIters, mask], [0, 1, 8], [3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findEssentialMat_2"] = "OpenCV findEssentialMat_2"
NODE_CLASS_MAPPINGS	["findEssentialMat_2"] = cv2_findEssentialMat_2

class cv2_findEssentialMat_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"focal"	: ("FLOAT",),
				"pp"	: ("STRING",),
				"method"	: ("INT",),
				"prob"	: ("FLOAT",),
				"threshold"	: ("FLOAT",),
				"maxIters"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, focal, pp, method, prob, threshold, maxIters, mask=None):
		ret = apply_function(cv2.findEssentialMat, [points1, points2, focal, pp, method, prob, threshold, maxIters, mask], [0, 1, 8], [3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findEssentialMat_3"] = "OpenCV findEssentialMat_3"
NODE_CLASS_MAPPINGS	["findEssentialMat_3"] = cv2_findEssentialMat_3

class cv2_findEssentialMat_4:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"cameraMatrix1"	: ("NPARRAY",),
				"distCoeffs1"	: ("NPARRAY",),
				"cameraMatrix2"	: ("NPARRAY",),
				"distCoeffs2"	: ("NPARRAY",),
				"method"	: ("INT",),
				"prob"	: ("FLOAT",),
				"threshold"	: ("FLOAT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, method, prob, threshold, mask=None):
		ret = apply_function(cv2.findEssentialMat, [points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, method, prob, threshold, mask], [0, 1, 2, 3, 4, 5, 9], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findEssentialMat_4"] = "OpenCV findEssentialMat_4"
NODE_CLASS_MAPPINGS	["findEssentialMat_4"] = cv2_findEssentialMat_4

class cv2_findEssentialMat_5:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"cameraMatrix1"	: ("NPARRAY",),
				"distCoeffs1"	: ("NPARRAY",),
				"cameraMatrix2"	: ("NPARRAY",),
				"distCoeffs2"	: ("NPARRAY",),
				"method"	: ("INT",),
				"prob"	: ("FLOAT",),
				"threshold"	: ("FLOAT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, method, prob, threshold, mask=None):
		ret = apply_function(cv2.findEssentialMat, [points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, method, prob, threshold, mask], [0, 1, 2, 3, 4, 5, 9], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findEssentialMat_5"] = "OpenCV findEssentialMat_5"
NODE_CLASS_MAPPINGS	["findEssentialMat_5"] = cv2_findEssentialMat_5

class cv2_findFundamentalMat_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"method"	: ("INT",),
				"ransacReprojThreshold"	: ("FLOAT",),
				"confidence"	: ("FLOAT",),
				"maxIters"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, method, ransacReprojThreshold, confidence, maxIters, mask=None):
		ret = apply_function(cv2.findFundamentalMat, [points1, points2, method, ransacReprojThreshold, confidence, maxIters, mask], [0, 1, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findFundamentalMat_0"] = "OpenCV findFundamentalMat_0"
NODE_CLASS_MAPPINGS	["findFundamentalMat_0"] = cv2_findFundamentalMat_0

class cv2_findFundamentalMat_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"method"	: ("INT",),
				"ransacReprojThreshold"	: ("FLOAT",),
				"confidence"	: ("FLOAT",),
				"maxIters"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, method, ransacReprojThreshold, confidence, maxIters, mask=None):
		ret = apply_function(cv2.findFundamentalMat, [points1, points2, method, ransacReprojThreshold, confidence, maxIters, mask], [0, 1, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findFundamentalMat_1"] = "OpenCV findFundamentalMat_1"
NODE_CLASS_MAPPINGS	["findFundamentalMat_1"] = cv2_findFundamentalMat_1

class cv2_findFundamentalMat_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"method"	: ("INT",),
				"ransacReprojThreshold"	: ("FLOAT",),
				"confidence"	: ("FLOAT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, method, ransacReprojThreshold, confidence, mask=None):
		ret = apply_function(cv2.findFundamentalMat, [points1, points2, method, ransacReprojThreshold, confidence, mask], [0, 1, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findFundamentalMat_2"] = "OpenCV findFundamentalMat_2"
NODE_CLASS_MAPPINGS	["findFundamentalMat_2"] = cv2_findFundamentalMat_2

class cv2_findFundamentalMat_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"method"	: ("INT",),
				"ransacReprojThreshold"	: ("FLOAT",),
				"confidence"	: ("FLOAT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, method, ransacReprojThreshold, confidence, mask=None):
		ret = apply_function(cv2.findFundamentalMat, [points1, points2, method, ransacReprojThreshold, confidence, mask], [0, 1, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findFundamentalMat_3"] = "OpenCV findFundamentalMat_3"
NODE_CLASS_MAPPINGS	["findFundamentalMat_3"] = cv2_findFundamentalMat_3

class cv2_findHomography_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"srcPoints"	: ("NPARRAY",),
				"dstPoints"	: ("NPARRAY",),
				"method"	: ("INT",),
				"ransacReprojThreshold"	: ("FLOAT",),
				"maxIters"	: ("INT",),
				"confidence"	: ("FLOAT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, srcPoints, dstPoints, method, ransacReprojThreshold, maxIters, confidence, mask=None):
		ret = apply_function(cv2.findHomography, [srcPoints, dstPoints, method, ransacReprojThreshold, mask, maxIters, confidence], [0, 1, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findHomography_0"] = "OpenCV findHomography_0"
NODE_CLASS_MAPPINGS	["findHomography_0"] = cv2_findHomography_0

class cv2_findHomography_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"srcPoints"	: ("NPARRAY",),
				"dstPoints"	: ("NPARRAY",),
				"method"	: ("INT",),
				"ransacReprojThreshold"	: ("FLOAT",),
				"maxIters"	: ("INT",),
				"confidence"	: ("FLOAT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, srcPoints, dstPoints, method, ransacReprojThreshold, maxIters, confidence, mask=None):
		ret = apply_function(cv2.findHomography, [srcPoints, dstPoints, method, ransacReprojThreshold, mask, maxIters, confidence], [0, 1, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findHomography_1"] = "OpenCV findHomography_1"
NODE_CLASS_MAPPINGS	["findHomography_1"] = cv2_findHomography_1

class cv2_findNonZero_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"idx"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, idx=None):
		ret = apply_function(cv2.findNonZero, [src, idx], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findNonZero_0"] = "OpenCV findNonZero_0"
NODE_CLASS_MAPPINGS	["findNonZero_0"] = cv2_findNonZero_0

class cv2_findNonZero_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"idx"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, idx=None):
		ret = apply_function(cv2.findNonZero, [src, idx], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findNonZero_1"] = "OpenCV findNonZero_1"
NODE_CLASS_MAPPINGS	["findNonZero_1"] = cv2_findNonZero_1

class cv2_findTransformECC_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"templateImage"	: ("NPARRAY",),
				"inputImage"	: ("NPARRAY",),
				"warpMatrix"	: ("NPARRAY",),
				"motionType"	: ("INT",),
				"criteria"	: ("STRING",),
				"inputMask"	: ("NPARRAY",),
				"gaussFiltSize"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, templateImage, inputImage, warpMatrix, motionType, criteria, inputMask, gaussFiltSize):
		ret = apply_function(cv2.findTransformECC, [templateImage, inputImage, warpMatrix, motionType, criteria, inputMask, gaussFiltSize], [0, 1, 2, 5], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findTransformECC_0"] = "OpenCV findTransformECC_0"
NODE_CLASS_MAPPINGS	["findTransformECC_0"] = cv2_findTransformECC_0

class cv2_findTransformECC_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"templateImage"	: ("NPARRAY",),
				"inputImage"	: ("NPARRAY",),
				"warpMatrix"	: ("NPARRAY",),
				"motionType"	: ("INT",),
				"criteria"	: ("STRING",),
				"inputMask"	: ("NPARRAY",),
				"gaussFiltSize"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, templateImage, inputImage, warpMatrix, motionType, criteria, inputMask, gaussFiltSize):
		ret = apply_function(cv2.findTransformECC, [templateImage, inputImage, warpMatrix, motionType, criteria, inputMask, gaussFiltSize], [0, 1, 2, 5], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findTransformECC_1"] = "OpenCV findTransformECC_1"
NODE_CLASS_MAPPINGS	["findTransformECC_1"] = cv2_findTransformECC_1

class cv2_findTransformECC_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"templateImage"	: ("NPARRAY",),
				"inputImage"	: ("NPARRAY",),
				"warpMatrix"	: ("NPARRAY",),
				"motionType"	: ("INT",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				"inputMask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, templateImage, inputImage, warpMatrix, motionType, criteria, inputMask=None):
		ret = apply_function(cv2.findTransformECC, [templateImage, inputImage, warpMatrix, motionType, criteria, inputMask], [0, 1, 2, 5], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findTransformECC_2"] = "OpenCV findTransformECC_2"
NODE_CLASS_MAPPINGS	["findTransformECC_2"] = cv2_findTransformECC_2

class cv2_findTransformECC_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"templateImage"	: ("NPARRAY",),
				"inputImage"	: ("NPARRAY",),
				"warpMatrix"	: ("NPARRAY",),
				"motionType"	: ("INT",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				"inputMask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, templateImage, inputImage, warpMatrix, motionType, criteria, inputMask=None):
		ret = apply_function(cv2.findTransformECC, [templateImage, inputImage, warpMatrix, motionType, criteria, inputMask], [0, 1, 2, 5], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["findTransformECC_3"] = "OpenCV findTransformECC_3"
NODE_CLASS_MAPPINGS	["findTransformECC_3"] = cv2_findTransformECC_3

class cv2_fitEllipse_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points):
		ret = apply_function(cv2.fitEllipse, [points], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fitEllipse_0"] = "OpenCV fitEllipse_0"
NODE_CLASS_MAPPINGS	["fitEllipse_0"] = cv2_fitEllipse_0

class cv2_fitEllipse_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points):
		ret = apply_function(cv2.fitEllipse, [points], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fitEllipse_1"] = "OpenCV fitEllipse_1"
NODE_CLASS_MAPPINGS	["fitEllipse_1"] = cv2_fitEllipse_1

class cv2_fitEllipseAMS_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points):
		ret = apply_function(cv2.fitEllipseAMS, [points], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fitEllipseAMS_0"] = "OpenCV fitEllipseAMS_0"
NODE_CLASS_MAPPINGS	["fitEllipseAMS_0"] = cv2_fitEllipseAMS_0

class cv2_fitEllipseAMS_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points):
		ret = apply_function(cv2.fitEllipseAMS, [points], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fitEllipseAMS_1"] = "OpenCV fitEllipseAMS_1"
NODE_CLASS_MAPPINGS	["fitEllipseAMS_1"] = cv2_fitEllipseAMS_1

class cv2_fitEllipseDirect_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points):
		ret = apply_function(cv2.fitEllipseDirect, [points], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fitEllipseDirect_0"] = "OpenCV fitEllipseDirect_0"
NODE_CLASS_MAPPINGS	["fitEllipseDirect_0"] = cv2_fitEllipseDirect_0

class cv2_fitEllipseDirect_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points):
		ret = apply_function(cv2.fitEllipseDirect, [points], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fitEllipseDirect_1"] = "OpenCV fitEllipseDirect_1"
NODE_CLASS_MAPPINGS	["fitEllipseDirect_1"] = cv2_fitEllipseDirect_1

class cv2_fitLine_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
				"distType"	: ("INT",),
				"param"	: ("FLOAT",),
				"reps"	: ("FLOAT",),
				"aeps"	: ("FLOAT",),
			},
			'optional': {
				"line"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points, distType, param, reps, aeps, line=None):
		ret = apply_function(cv2.fitLine, [points, distType, param, reps, aeps, line], [0, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fitLine_0"] = "OpenCV fitLine_0"
NODE_CLASS_MAPPINGS	["fitLine_0"] = cv2_fitLine_0

class cv2_fitLine_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
				"distType"	: ("INT",),
				"param"	: ("FLOAT",),
				"reps"	: ("FLOAT",),
				"aeps"	: ("FLOAT",),
			},
			'optional': {
				"line"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points, distType, param, reps, aeps, line=None):
		ret = apply_function(cv2.fitLine, [points, distType, param, reps, aeps, line], [0, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["fitLine_1"] = "OpenCV fitLine_1"
NODE_CLASS_MAPPINGS	["fitLine_1"] = cv2_fitLine_1

class cv2_flip_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flipCode"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flipCode, dst=None):
		ret = apply_function(cv2.flip, [src, flipCode, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["flip_0"] = "OpenCV flip_0"
NODE_CLASS_MAPPINGS	["flip_0"] = cv2_flip_0

class cv2_flip_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flipCode"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flipCode, dst=None):
		ret = apply_function(cv2.flip, [src, flipCode, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["flip_1"] = "OpenCV flip_1"
NODE_CLASS_MAPPINGS	["flip_1"] = cv2_flip_1

class cv2_flipND_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"axis"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, axis, dst=None):
		ret = apply_function(cv2.flipND, [src, axis, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["flipND_0"] = "OpenCV flipND_0"
NODE_CLASS_MAPPINGS	["flipND_0"] = cv2_flipND_0

class cv2_flipND_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"axis"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, axis, dst=None):
		ret = apply_function(cv2.flipND, [src, axis, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["flipND_1"] = "OpenCV flipND_1"
NODE_CLASS_MAPPINGS	["flipND_1"] = cv2_flipND_1

class cv2_floodFill_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"seedPoint"	: ("STRING",),
				"newVal"	: ("STRING",),
				"loDiff"	: ("STRING",),
				"upDiff"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "STRING",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, seedPoint, newVal, loDiff, upDiff, flags, mask=None):
		ret = apply_function(cv2.floodFill, [image, mask, seedPoint, newVal, loDiff, upDiff, flags], [0, 1], [2, 3, 4, 5])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["floodFill_0"] = "OpenCV floodFill_0"
NODE_CLASS_MAPPINGS	["floodFill_0"] = cv2_floodFill_0

class cv2_floodFill_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"seedPoint"	: ("STRING",),
				"newVal"	: ("STRING",),
				"loDiff"	: ("STRING",),
				"upDiff"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "STRING",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, seedPoint, newVal, loDiff, upDiff, flags, mask=None):
		ret = apply_function(cv2.floodFill, [image, mask, seedPoint, newVal, loDiff, upDiff, flags], [0, 1], [2, 3, 4, 5])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["floodFill_1"] = "OpenCV floodFill_1"
NODE_CLASS_MAPPINGS	["floodFill_1"] = cv2_floodFill_1

class cv2_gemm_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"src3"	: ("NPARRAY",),
				"beta"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, alpha, src3, beta, flags, dst=None):
		ret = apply_function(cv2.gemm, [src1, src2, alpha, src3, beta, dst, flags], [0, 1, 3, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["gemm_0"] = "OpenCV gemm_0"
NODE_CLASS_MAPPINGS	["gemm_0"] = cv2_gemm_0

class cv2_gemm_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"src3"	: ("NPARRAY",),
				"beta"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, alpha, src3, beta, flags, dst=None):
		ret = apply_function(cv2.gemm, [src1, src2, alpha, src3, beta, dst, flags], [0, 1, 3, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["gemm_1"] = "OpenCV gemm_1"
NODE_CLASS_MAPPINGS	["gemm_1"] = cv2_gemm_1

class cv2_getAffineTransform_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst):
		ret = apply_function(cv2.getAffineTransform, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getAffineTransform_0"] = "OpenCV getAffineTransform_0"
NODE_CLASS_MAPPINGS	["getAffineTransform_0"] = cv2_getAffineTransform_0

class cv2_getAffineTransform_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst):
		ret = apply_function(cv2.getAffineTransform, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getAffineTransform_1"] = "OpenCV getAffineTransform_1"
NODE_CLASS_MAPPINGS	["getAffineTransform_1"] = cv2_getAffineTransform_1

class cv2_getBuildInformation_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("string",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getBuildInformation, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getBuildInformation_0"] = "OpenCV getBuildInformation_0"
NODE_CLASS_MAPPINGS	["getBuildInformation_0"] = cv2_getBuildInformation_0

class cv2_getCPUFeaturesLine_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("string",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getCPUFeaturesLine, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getCPUFeaturesLine_0"] = "OpenCV getCPUFeaturesLine_0"
NODE_CLASS_MAPPINGS	["getCPUFeaturesLine_0"] = cv2_getCPUFeaturesLine_0

class cv2_getCPUTickCount_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getCPUTickCount, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getCPUTickCount_0"] = "OpenCV getCPUTickCount_0"
NODE_CLASS_MAPPINGS	["getCPUTickCount_0"] = cv2_getCPUTickCount_0

class cv2_getDefaultAlgorithmHint_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getDefaultAlgorithmHint, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getDefaultAlgorithmHint_0"] = "OpenCV getDefaultAlgorithmHint_0"
NODE_CLASS_MAPPINGS	["getDefaultAlgorithmHint_0"] = cv2_getDefaultAlgorithmHint_0

class cv2_getDefaultNewCameraMatrix_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix"	: ("NPARRAY",),
				"imgsize"	: ("STRING",),
				"centerPrincipalPoint"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix, imgsize, centerPrincipalPoint):
		ret = apply_function(cv2.getDefaultNewCameraMatrix, [cameraMatrix, imgsize, centerPrincipalPoint], [0], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getDefaultNewCameraMatrix_0"] = "OpenCV getDefaultNewCameraMatrix_0"
NODE_CLASS_MAPPINGS	["getDefaultNewCameraMatrix_0"] = cv2_getDefaultNewCameraMatrix_0

class cv2_getDefaultNewCameraMatrix_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix"	: ("NPARRAY",),
				"imgsize"	: ("STRING",),
				"centerPrincipalPoint"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix, imgsize, centerPrincipalPoint):
		ret = apply_function(cv2.getDefaultNewCameraMatrix, [cameraMatrix, imgsize, centerPrincipalPoint], [0], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getDefaultNewCameraMatrix_1"] = "OpenCV getDefaultNewCameraMatrix_1"
NODE_CLASS_MAPPINGS	["getDefaultNewCameraMatrix_1"] = cv2_getDefaultNewCameraMatrix_1

class cv2_getDerivKernels_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"dx"	: ("INT",),
				"dy"	: ("INT",),
				"ksize"	: ("INT",),
				"normalize"	: ("BOOLEAN",),
				"ktype"	: ("INT",),
			},
			'optional': {
				"kx"	: ("NPARRAY",),
				"ky"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, dx, dy, ksize, normalize, ktype, kx=None, ky=None):
		ret = apply_function(cv2.getDerivKernels, [dx, dy, ksize, kx, ky, normalize, ktype], [3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getDerivKernels_0"] = "OpenCV getDerivKernels_0"
NODE_CLASS_MAPPINGS	["getDerivKernels_0"] = cv2_getDerivKernels_0

class cv2_getDerivKernels_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"dx"	: ("INT",),
				"dy"	: ("INT",),
				"ksize"	: ("INT",),
				"normalize"	: ("BOOLEAN",),
				"ktype"	: ("INT",),
			},
			'optional': {
				"kx"	: ("NPARRAY",),
				"ky"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, dx, dy, ksize, normalize, ktype, kx=None, ky=None):
		ret = apply_function(cv2.getDerivKernels, [dx, dy, ksize, kx, ky, normalize, ktype], [3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getDerivKernels_1"] = "OpenCV getDerivKernels_1"
NODE_CLASS_MAPPINGS	["getDerivKernels_1"] = cv2_getDerivKernels_1

class cv2_getFontScaleFromHeight_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"fontFace"	: ("INT",),
				"pixelHeight"	: ("INT",),
				"thickness"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, fontFace, pixelHeight, thickness):
		ret = apply_function(cv2.getFontScaleFromHeight, [fontFace, pixelHeight, thickness], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getFontScaleFromHeight_0"] = "OpenCV getFontScaleFromHeight_0"
NODE_CLASS_MAPPINGS	["getFontScaleFromHeight_0"] = cv2_getFontScaleFromHeight_0

class cv2_getGaborKernel_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"ksize"	: ("STRING",),
				"sigma"	: ("FLOAT",),
				"theta"	: ("FLOAT",),
				"lambd"	: ("FLOAT",),
				"gamma"	: ("FLOAT",),
				"psi"	: ("FLOAT",),
				"ktype"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ksize, sigma, theta, lambd, gamma, psi, ktype):
		ret = apply_function(cv2.getGaborKernel, [ksize, sigma, theta, lambd, gamma, psi, ktype], [], [0])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getGaborKernel_0"] = "OpenCV getGaborKernel_0"
NODE_CLASS_MAPPINGS	["getGaborKernel_0"] = cv2_getGaborKernel_0

class cv2_getGaussianKernel_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"ksize"	: ("INT",),
				"sigma"	: ("FLOAT",),
				"ktype"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ksize, sigma, ktype):
		ret = apply_function(cv2.getGaussianKernel, [ksize, sigma, ktype], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getGaussianKernel_0"] = "OpenCV getGaussianKernel_0"
NODE_CLASS_MAPPINGS	["getGaussianKernel_0"] = cv2_getGaussianKernel_0

class cv2_getHardwareFeatureName_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"feature"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("string",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, feature):
		ret = apply_function(cv2.getHardwareFeatureName, [feature], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getHardwareFeatureName_0"] = "OpenCV getHardwareFeatureName_0"
NODE_CLASS_MAPPINGS	["getHardwareFeatureName_0"] = cv2_getHardwareFeatureName_0

class cv2_getLogLevel_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getLogLevel, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getLogLevel_0"] = "OpenCV getLogLevel_0"
NODE_CLASS_MAPPINGS	["getLogLevel_0"] = cv2_getLogLevel_0

class cv2_getNumThreads_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getNumThreads, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getNumThreads_0"] = "OpenCV getNumThreads_0"
NODE_CLASS_MAPPINGS	["getNumThreads_0"] = cv2_getNumThreads_0

class cv2_getNumberOfCPUs_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getNumberOfCPUs, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getNumberOfCPUs_0"] = "OpenCV getNumberOfCPUs_0"
NODE_CLASS_MAPPINGS	["getNumberOfCPUs_0"] = cv2_getNumberOfCPUs_0

class cv2_getOptimalDFTSize_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"vecsize"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, vecsize):
		ret = apply_function(cv2.getOptimalDFTSize, [vecsize], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getOptimalDFTSize_0"] = "OpenCV getOptimalDFTSize_0"
NODE_CLASS_MAPPINGS	["getOptimalDFTSize_0"] = cv2_getOptimalDFTSize_0

class cv2_getOptimalNewCameraMatrix_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"imageSize"	: ("STRING",),
				"alpha"	: ("FLOAT",),
				"newImgSize"	: ("STRING",),
				"centerPrincipalPoint"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY", "STRING",)
	RETURN_NAMES	= ("nparray", "literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix, distCoeffs, imageSize, alpha, newImgSize, centerPrincipalPoint):
		ret = apply_function(cv2.getOptimalNewCameraMatrix, [cameraMatrix, distCoeffs, imageSize, alpha, newImgSize, centerPrincipalPoint], [0, 1], [2, 4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getOptimalNewCameraMatrix_0"] = "OpenCV getOptimalNewCameraMatrix_0"
NODE_CLASS_MAPPINGS	["getOptimalNewCameraMatrix_0"] = cv2_getOptimalNewCameraMatrix_0

class cv2_getOptimalNewCameraMatrix_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"imageSize"	: ("STRING",),
				"alpha"	: ("FLOAT",),
				"newImgSize"	: ("STRING",),
				"centerPrincipalPoint"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY", "STRING",)
	RETURN_NAMES	= ("nparray", "literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix, distCoeffs, imageSize, alpha, newImgSize, centerPrincipalPoint):
		ret = apply_function(cv2.getOptimalNewCameraMatrix, [cameraMatrix, distCoeffs, imageSize, alpha, newImgSize, centerPrincipalPoint], [0, 1], [2, 4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getOptimalNewCameraMatrix_1"] = "OpenCV getOptimalNewCameraMatrix_1"
NODE_CLASS_MAPPINGS	["getOptimalNewCameraMatrix_1"] = cv2_getOptimalNewCameraMatrix_1

class cv2_getPerspectiveTransform_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"solveMethod"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, solveMethod):
		ret = apply_function(cv2.getPerspectiveTransform, [src, dst, solveMethod], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getPerspectiveTransform_0"] = "OpenCV getPerspectiveTransform_0"
NODE_CLASS_MAPPINGS	["getPerspectiveTransform_0"] = cv2_getPerspectiveTransform_0

class cv2_getPerspectiveTransform_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"solveMethod"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, solveMethod):
		ret = apply_function(cv2.getPerspectiveTransform, [src, dst, solveMethod], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getPerspectiveTransform_1"] = "OpenCV getPerspectiveTransform_1"
NODE_CLASS_MAPPINGS	["getPerspectiveTransform_1"] = cv2_getPerspectiveTransform_1

class cv2_getRectSubPix_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patchSize"	: ("STRING",),
				"center"	: ("STRING",),
				"patchType"	: ("INT",),
			},
			'optional': {
				"patch"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patchSize, center, patchType, patch=None):
		ret = apply_function(cv2.getRectSubPix, [image, patchSize, center, patch, patchType], [0, 3], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getRectSubPix_0"] = "OpenCV getRectSubPix_0"
NODE_CLASS_MAPPINGS	["getRectSubPix_0"] = cv2_getRectSubPix_0

class cv2_getRectSubPix_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"patchSize"	: ("STRING",),
				"center"	: ("STRING",),
				"patchType"	: ("INT",),
			},
			'optional': {
				"patch"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, patchSize, center, patchType, patch=None):
		ret = apply_function(cv2.getRectSubPix, [image, patchSize, center, patch, patchType], [0, 3], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getRectSubPix_1"] = "OpenCV getRectSubPix_1"
NODE_CLASS_MAPPINGS	["getRectSubPix_1"] = cv2_getRectSubPix_1

class cv2_getRotationMatrix2D_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"center"	: ("STRING",),
				"angle"	: ("FLOAT",),
				"scale"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, center, angle, scale):
		ret = apply_function(cv2.getRotationMatrix2D, [center, angle, scale], [], [0])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getRotationMatrix2D_0"] = "OpenCV getRotationMatrix2D_0"
NODE_CLASS_MAPPINGS	["getRotationMatrix2D_0"] = cv2_getRotationMatrix2D_0

class cv2_getStructuringElement_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"shape"	: ("INT",),
				"ksize"	: ("STRING",),
				"anchor"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, shape, ksize, anchor):
		ret = apply_function(cv2.getStructuringElement, [shape, ksize, anchor], [], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getStructuringElement_0"] = "OpenCV getStructuringElement_0"
NODE_CLASS_MAPPINGS	["getStructuringElement_0"] = cv2_getStructuringElement_0

class cv2_getTextSize_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"text"	: ("STRING",),
				"fontFace"	: ("INT",),
				"fontScale"	: ("FLOAT",),
				"thickness"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING", "INT",)
	RETURN_NAMES	= ("literal", "int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, text, fontFace, fontScale, thickness):
		ret = apply_function(cv2.getTextSize, [text, fontFace, fontScale, thickness], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getTextSize_0"] = "OpenCV getTextSize_0"
NODE_CLASS_MAPPINGS	["getTextSize_0"] = cv2_getTextSize_0

class cv2_getThreadNum_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getThreadNum, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getThreadNum_0"] = "OpenCV getThreadNum_0"
NODE_CLASS_MAPPINGS	["getThreadNum_0"] = cv2_getThreadNum_0

class cv2_getTickCount_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getTickCount, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getTickCount_0"] = "OpenCV getTickCount_0"
NODE_CLASS_MAPPINGS	["getTickCount_0"] = cv2_getTickCount_0

class cv2_getTickFrequency_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getTickFrequency, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getTickFrequency_0"] = "OpenCV getTickFrequency_0"
NODE_CLASS_MAPPINGS	["getTickFrequency_0"] = cv2_getTickFrequency_0

class cv2_getTrackbarPos_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"trackbarname"	: ("STRING",),
				"winname"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, trackbarname, winname):
		ret = apply_function(cv2.getTrackbarPos, [trackbarname, winname], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getTrackbarPos_0"] = "OpenCV getTrackbarPos_0"
NODE_CLASS_MAPPINGS	["getTrackbarPos_0"] = cv2_getTrackbarPos_0

class cv2_getValidDisparityROI_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"roi1"	: ("STRING",),
				"roi2"	: ("STRING",),
				"minDisparity"	: ("INT",),
				"numberOfDisparities"	: ("INT",),
				"blockSize"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, roi1, roi2, minDisparity, numberOfDisparities, blockSize):
		ret = apply_function(cv2.getValidDisparityROI, [roi1, roi2, minDisparity, numberOfDisparities, blockSize], [], [0, 1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getValidDisparityROI_0"] = "OpenCV getValidDisparityROI_0"
NODE_CLASS_MAPPINGS	["getValidDisparityROI_0"] = cv2_getValidDisparityROI_0

class cv2_getVersionMajor_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getVersionMajor, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getVersionMajor_0"] = "OpenCV getVersionMajor_0"
NODE_CLASS_MAPPINGS	["getVersionMajor_0"] = cv2_getVersionMajor_0

class cv2_getVersionMinor_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getVersionMinor, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getVersionMinor_0"] = "OpenCV getVersionMinor_0"
NODE_CLASS_MAPPINGS	["getVersionMinor_0"] = cv2_getVersionMinor_0

class cv2_getVersionRevision_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getVersionRevision, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getVersionRevision_0"] = "OpenCV getVersionRevision_0"
NODE_CLASS_MAPPINGS	["getVersionRevision_0"] = cv2_getVersionRevision_0

class cv2_getVersionString_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("string",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.getVersionString, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getVersionString_0"] = "OpenCV getVersionString_0"
NODE_CLASS_MAPPINGS	["getVersionString_0"] = cv2_getVersionString_0

class cv2_getWindowImageRect_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname):
		ret = apply_function(cv2.getWindowImageRect, [winname], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getWindowImageRect_0"] = "OpenCV getWindowImageRect_0"
NODE_CLASS_MAPPINGS	["getWindowImageRect_0"] = cv2_getWindowImageRect_0

class cv2_getWindowProperty_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"prop_id"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, prop_id):
		ret = apply_function(cv2.getWindowProperty, [winname, prop_id], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["getWindowProperty_0"] = "OpenCV getWindowProperty_0"
NODE_CLASS_MAPPINGS	["getWindowProperty_0"] = cv2_getWindowProperty_0

class cv2_goodFeaturesToTrack_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"maxCorners"	: ("INT",),
				"qualityLevel"	: ("FLOAT",),
				"minDistance"	: ("FLOAT",),
				"blockSize"	: ("INT",),
				"useHarrisDetector"	: ("BOOLEAN",),
				"k"	: ("FLOAT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k, corners=None, mask=None):
		ret = apply_function(cv2.goodFeaturesToTrack, [image, maxCorners, qualityLevel, minDistance, corners, mask, blockSize, useHarrisDetector, k], [0, 4, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["goodFeaturesToTrack_0"] = "OpenCV goodFeaturesToTrack_0"
NODE_CLASS_MAPPINGS	["goodFeaturesToTrack_0"] = cv2_goodFeaturesToTrack_0

class cv2_goodFeaturesToTrack_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"maxCorners"	: ("INT",),
				"qualityLevel"	: ("FLOAT",),
				"minDistance"	: ("FLOAT",),
				"blockSize"	: ("INT",),
				"useHarrisDetector"	: ("BOOLEAN",),
				"k"	: ("FLOAT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k, corners=None, mask=None):
		ret = apply_function(cv2.goodFeaturesToTrack, [image, maxCorners, qualityLevel, minDistance, corners, mask, blockSize, useHarrisDetector, k], [0, 4, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["goodFeaturesToTrack_1"] = "OpenCV goodFeaturesToTrack_1"
NODE_CLASS_MAPPINGS	["goodFeaturesToTrack_1"] = cv2_goodFeaturesToTrack_1

class cv2_goodFeaturesToTrack_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"maxCorners"	: ("INT",),
				"qualityLevel"	: ("FLOAT",),
				"minDistance"	: ("FLOAT",),
				"mask"	: ("NPARRAY",),
				"blockSize"	: ("INT",),
				"gradientSize"	: ("INT",),
				"useHarrisDetector"	: ("BOOLEAN",),
				"k"	: ("FLOAT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, maxCorners, qualityLevel, minDistance, mask, blockSize, gradientSize, useHarrisDetector, k, corners=None):
		ret = apply_function(cv2.goodFeaturesToTrack, [image, maxCorners, qualityLevel, minDistance, mask, blockSize, gradientSize, corners, useHarrisDetector, k], [0, 4, 7], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["goodFeaturesToTrack_2"] = "OpenCV goodFeaturesToTrack_2"
NODE_CLASS_MAPPINGS	["goodFeaturesToTrack_2"] = cv2_goodFeaturesToTrack_2

class cv2_goodFeaturesToTrack_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"maxCorners"	: ("INT",),
				"qualityLevel"	: ("FLOAT",),
				"minDistance"	: ("FLOAT",),
				"mask"	: ("NPARRAY",),
				"blockSize"	: ("INT",),
				"gradientSize"	: ("INT",),
				"useHarrisDetector"	: ("BOOLEAN",),
				"k"	: ("FLOAT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, maxCorners, qualityLevel, minDistance, mask, blockSize, gradientSize, useHarrisDetector, k, corners=None):
		ret = apply_function(cv2.goodFeaturesToTrack, [image, maxCorners, qualityLevel, minDistance, mask, blockSize, gradientSize, corners, useHarrisDetector, k], [0, 4, 7], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["goodFeaturesToTrack_3"] = "OpenCV goodFeaturesToTrack_3"
NODE_CLASS_MAPPINGS	["goodFeaturesToTrack_3"] = cv2_goodFeaturesToTrack_3

class cv2_goodFeaturesToTrackWithQuality_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"maxCorners"	: ("INT",),
				"qualityLevel"	: ("FLOAT",),
				"minDistance"	: ("FLOAT",),
				"mask"	: ("NPARRAY",),
				"blockSize"	: ("INT",),
				"gradientSize"	: ("INT",),
				"useHarrisDetector"	: ("BOOLEAN",),
				"k"	: ("FLOAT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
				"cornersQuality"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, maxCorners, qualityLevel, minDistance, mask, blockSize, gradientSize, useHarrisDetector, k, corners=None, cornersQuality=None):
		ret = apply_function(cv2.goodFeaturesToTrackWithQuality, [image, maxCorners, qualityLevel, minDistance, mask, corners, cornersQuality, blockSize, gradientSize, useHarrisDetector, k], [0, 4, 5, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["goodFeaturesToTrackWithQuality_0"] = "OpenCV goodFeaturesToTrackWithQuality_0"
NODE_CLASS_MAPPINGS	["goodFeaturesToTrackWithQuality_0"] = cv2_goodFeaturesToTrackWithQuality_0

class cv2_goodFeaturesToTrackWithQuality_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"maxCorners"	: ("INT",),
				"qualityLevel"	: ("FLOAT",),
				"minDistance"	: ("FLOAT",),
				"mask"	: ("NPARRAY",),
				"blockSize"	: ("INT",),
				"gradientSize"	: ("INT",),
				"useHarrisDetector"	: ("BOOLEAN",),
				"k"	: ("FLOAT",),
			},
			'optional': {
				"corners"	: ("NPARRAY",),
				"cornersQuality"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, maxCorners, qualityLevel, minDistance, mask, blockSize, gradientSize, useHarrisDetector, k, corners=None, cornersQuality=None):
		ret = apply_function(cv2.goodFeaturesToTrackWithQuality, [image, maxCorners, qualityLevel, minDistance, mask, corners, cornersQuality, blockSize, gradientSize, useHarrisDetector, k], [0, 4, 5, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["goodFeaturesToTrackWithQuality_1"] = "OpenCV goodFeaturesToTrackWithQuality_1"
NODE_CLASS_MAPPINGS	["goodFeaturesToTrackWithQuality_1"] = cv2_goodFeaturesToTrackWithQuality_1

class cv2_grabCut_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"rect"	: ("STRING",),
				"bgdModel"	: ("NPARRAY",),
				"fgdModel"	: ("NPARRAY",),
				"iterCount"	: ("INT",),
				"mode"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, mask, rect, bgdModel, fgdModel, iterCount, mode):
		ret = apply_function(cv2.grabCut, [img, mask, rect, bgdModel, fgdModel, iterCount, mode], [0, 1, 3, 4], [2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["grabCut_0"] = "OpenCV grabCut_0"
NODE_CLASS_MAPPINGS	["grabCut_0"] = cv2_grabCut_0

class cv2_grabCut_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"rect"	: ("STRING",),
				"bgdModel"	: ("NPARRAY",),
				"fgdModel"	: ("NPARRAY",),
				"iterCount"	: ("INT",),
				"mode"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, mask, rect, bgdModel, fgdModel, iterCount, mode):
		ret = apply_function(cv2.grabCut, [img, mask, rect, bgdModel, fgdModel, iterCount, mode], [0, 1, 3, 4], [2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["grabCut_1"] = "OpenCV grabCut_1"
NODE_CLASS_MAPPINGS	["grabCut_1"] = cv2_grabCut_1

class cv2_hasNonZero_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src):
		ret = apply_function(cv2.hasNonZero, [src], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["hasNonZero_0"] = "OpenCV hasNonZero_0"
NODE_CLASS_MAPPINGS	["hasNonZero_0"] = cv2_hasNonZero_0

class cv2_hasNonZero_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src):
		ret = apply_function(cv2.hasNonZero, [src], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["hasNonZero_1"] = "OpenCV hasNonZero_1"
NODE_CLASS_MAPPINGS	["hasNonZero_1"] = cv2_hasNonZero_1

class cv2_haveImageReader_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"filename"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, filename):
		ret = apply_function(cv2.haveImageReader, [filename], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["haveImageReader_0"] = "OpenCV haveImageReader_0"
NODE_CLASS_MAPPINGS	["haveImageReader_0"] = cv2_haveImageReader_0

class cv2_haveImageWriter_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"filename"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, filename):
		ret = apply_function(cv2.haveImageWriter, [filename], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["haveImageWriter_0"] = "OpenCV haveImageWriter_0"
NODE_CLASS_MAPPINGS	["haveImageWriter_0"] = cv2_haveImageWriter_0

class cv2_haveOpenVX_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.haveOpenVX, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["haveOpenVX_0"] = "OpenCV haveOpenVX_0"
NODE_CLASS_MAPPINGS	["haveOpenVX_0"] = cv2_haveOpenVX_0

class cv2_idct_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, dst=None):
		ret = apply_function(cv2.idct, [src, dst, flags], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["idct_0"] = "OpenCV idct_0"
NODE_CLASS_MAPPINGS	["idct_0"] = cv2_idct_0

class cv2_idct_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, dst=None):
		ret = apply_function(cv2.idct, [src, dst, flags], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["idct_1"] = "OpenCV idct_1"
NODE_CLASS_MAPPINGS	["idct_1"] = cv2_idct_1

class cv2_idft_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"nonzeroRows"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, nonzeroRows, dst=None):
		ret = apply_function(cv2.idft, [src, dst, flags, nonzeroRows], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["idft_0"] = "OpenCV idft_0"
NODE_CLASS_MAPPINGS	["idft_0"] = cv2_idft_0

class cv2_idft_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"nonzeroRows"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, nonzeroRows, dst=None):
		ret = apply_function(cv2.idft, [src, dst, flags, nonzeroRows], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["idft_1"] = "OpenCV idft_1"
NODE_CLASS_MAPPINGS	["idft_1"] = cv2_idft_1

class cv2_illuminationChange_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"beta"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask, alpha, beta, dst=None):
		ret = apply_function(cv2.illuminationChange, [src, mask, dst, alpha, beta], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["illuminationChange_0"] = "OpenCV illuminationChange_0"
NODE_CLASS_MAPPINGS	["illuminationChange_0"] = cv2_illuminationChange_0

class cv2_illuminationChange_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"beta"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask, alpha, beta, dst=None):
		ret = apply_function(cv2.illuminationChange, [src, mask, dst, alpha, beta], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["illuminationChange_1"] = "OpenCV illuminationChange_1"
NODE_CLASS_MAPPINGS	["illuminationChange_1"] = cv2_illuminationChange_1

class cv2_imcount_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"filename"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, filename, flags):
		ret = apply_function(cv2.imcount, [filename, flags], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["imcount_0"] = "OpenCV imcount_0"
NODE_CLASS_MAPPINGS	["imcount_0"] = cv2_imcount_0

class cv2_imdecode_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"buf"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, buf, flags):
		ret = apply_function(cv2.imdecode, [buf, flags], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["imdecode_0"] = "OpenCV imdecode_0"
NODE_CLASS_MAPPINGS	["imdecode_0"] = cv2_imdecode_0

class cv2_imdecode_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"buf"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, buf, flags):
		ret = apply_function(cv2.imdecode, [buf, flags], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["imdecode_1"] = "OpenCV imdecode_1"
NODE_CLASS_MAPPINGS	["imdecode_1"] = cv2_imdecode_1

class cv2_imread_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"filename"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, filename, flags):
		ret = apply_function(cv2.imread, [filename, flags], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["imread_0"] = "OpenCV imread_0"
NODE_CLASS_MAPPINGS	["imread_0"] = cv2_imread_0

class cv2_imread_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"filename"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, filename, flags, dst=None):
		ret = apply_function(cv2.imread, [filename, dst, flags], [1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["imread_1"] = "OpenCV imread_1"
NODE_CLASS_MAPPINGS	["imread_1"] = cv2_imread_1

class cv2_imread_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"filename"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, filename, flags, dst=None):
		ret = apply_function(cv2.imread, [filename, dst, flags], [1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["imread_2"] = "OpenCV imread_2"
NODE_CLASS_MAPPINGS	["imread_2"] = cv2_imread_2

class cv2_imshow_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"mat"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, mat):
		ret = apply_function(cv2.imshow, [winname, mat], [1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["imshow_0"] = "OpenCV imshow_0"
NODE_CLASS_MAPPINGS	["imshow_0"] = cv2_imshow_0

class cv2_imshow_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"mat"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, mat):
		ret = apply_function(cv2.imshow, [winname, mat], [1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["imshow_1"] = "OpenCV imshow_1"
NODE_CLASS_MAPPINGS	["imshow_1"] = cv2_imshow_1

class cv2_imshow_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"mat"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, mat):
		ret = apply_function(cv2.imshow, [winname, mat], [1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["imshow_2"] = "OpenCV imshow_2"
NODE_CLASS_MAPPINGS	["imshow_2"] = cv2_imshow_2

class cv2_inRange_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"lowerb"	: ("NPARRAY",),
				"upperb"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, lowerb, upperb, dst=None):
		ret = apply_function(cv2.inRange, [src, lowerb, upperb, dst], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["inRange_0"] = "OpenCV inRange_0"
NODE_CLASS_MAPPINGS	["inRange_0"] = cv2_inRange_0

class cv2_inRange_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"lowerb"	: ("NPARRAY",),
				"upperb"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, lowerb, upperb, dst=None):
		ret = apply_function(cv2.inRange, [src, lowerb, upperb, dst], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["inRange_1"] = "OpenCV inRange_1"
NODE_CLASS_MAPPINGS	["inRange_1"] = cv2_inRange_1

class cv2_initInverseRectificationMap_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"R"	: ("NPARRAY",),
				"newCameraMatrix"	: ("NPARRAY",),
				"size"	: ("STRING",),
				"m1type"	: ("INT",),
			},
			'optional': {
				"map1"	: ("NPARRAY",),
				"map2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1=None, map2=None):
		ret = apply_function(cv2.initInverseRectificationMap, [cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1, map2], [0, 1, 2, 3, 6, 7], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["initInverseRectificationMap_0"] = "OpenCV initInverseRectificationMap_0"
NODE_CLASS_MAPPINGS	["initInverseRectificationMap_0"] = cv2_initInverseRectificationMap_0

class cv2_initInverseRectificationMap_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"R"	: ("NPARRAY",),
				"newCameraMatrix"	: ("NPARRAY",),
				"size"	: ("STRING",),
				"m1type"	: ("INT",),
			},
			'optional': {
				"map1"	: ("NPARRAY",),
				"map2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1=None, map2=None):
		ret = apply_function(cv2.initInverseRectificationMap, [cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1, map2], [0, 1, 2, 3, 6, 7], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["initInverseRectificationMap_1"] = "OpenCV initInverseRectificationMap_1"
NODE_CLASS_MAPPINGS	["initInverseRectificationMap_1"] = cv2_initInverseRectificationMap_1

class cv2_initUndistortRectifyMap_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"R"	: ("NPARRAY",),
				"newCameraMatrix"	: ("NPARRAY",),
				"size"	: ("STRING",),
				"m1type"	: ("INT",),
			},
			'optional': {
				"map1"	: ("NPARRAY",),
				"map2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1=None, map2=None):
		ret = apply_function(cv2.initUndistortRectifyMap, [cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1, map2], [0, 1, 2, 3, 6, 7], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["initUndistortRectifyMap_0"] = "OpenCV initUndistortRectifyMap_0"
NODE_CLASS_MAPPINGS	["initUndistortRectifyMap_0"] = cv2_initUndistortRectifyMap_0

class cv2_initUndistortRectifyMap_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"R"	: ("NPARRAY",),
				"newCameraMatrix"	: ("NPARRAY",),
				"size"	: ("STRING",),
				"m1type"	: ("INT",),
			},
			'optional': {
				"map1"	: ("NPARRAY",),
				"map2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1=None, map2=None):
		ret = apply_function(cv2.initUndistortRectifyMap, [cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1, map2], [0, 1, 2, 3, 6, 7], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["initUndistortRectifyMap_1"] = "OpenCV initUndistortRectifyMap_1"
NODE_CLASS_MAPPINGS	["initUndistortRectifyMap_1"] = cv2_initUndistortRectifyMap_1

class cv2_inpaint_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"inpaintMask"	: ("NPARRAY",),
				"inpaintRadius"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, inpaintMask, inpaintRadius, flags, dst=None):
		ret = apply_function(cv2.inpaint, [src, inpaintMask, inpaintRadius, flags, dst], [0, 1, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["inpaint_0"] = "OpenCV inpaint_0"
NODE_CLASS_MAPPINGS	["inpaint_0"] = cv2_inpaint_0

class cv2_inpaint_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"inpaintMask"	: ("NPARRAY",),
				"inpaintRadius"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, inpaintMask, inpaintRadius, flags, dst=None):
		ret = apply_function(cv2.inpaint, [src, inpaintMask, inpaintRadius, flags, dst], [0, 1, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["inpaint_1"] = "OpenCV inpaint_1"
NODE_CLASS_MAPPINGS	["inpaint_1"] = cv2_inpaint_1

class cv2_insertChannel_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"coi"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, coi):
		ret = apply_function(cv2.insertChannel, [src, dst, coi], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["insertChannel_0"] = "OpenCV insertChannel_0"
NODE_CLASS_MAPPINGS	["insertChannel_0"] = cv2_insertChannel_0

class cv2_insertChannel_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"coi"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, coi):
		ret = apply_function(cv2.insertChannel, [src, dst, coi], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["insertChannel_1"] = "OpenCV insertChannel_1"
NODE_CLASS_MAPPINGS	["insertChannel_1"] = cv2_insertChannel_1

class cv2_integral_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sdepth"	: ("INT",),
			},
			'optional': {
				"sum"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sdepth, sum=None):
		ret = apply_function(cv2.integral, [src, sum, sdepth], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["integral_0"] = "OpenCV integral_0"
NODE_CLASS_MAPPINGS	["integral_0"] = cv2_integral_0

class cv2_integral_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sdepth"	: ("INT",),
			},
			'optional': {
				"sum"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sdepth, sum=None):
		ret = apply_function(cv2.integral, [src, sum, sdepth], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["integral_1"] = "OpenCV integral_1"
NODE_CLASS_MAPPINGS	["integral_1"] = cv2_integral_1

class cv2_integral2_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sdepth"	: ("INT",),
				"sqdepth"	: ("INT",),
			},
			'optional': {
				"sum"	: ("NPARRAY",),
				"sqsum"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sdepth, sqdepth, sum=None, sqsum=None):
		ret = apply_function(cv2.integral2, [src, sum, sqsum, sdepth, sqdepth], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["integral2_0"] = "OpenCV integral2_0"
NODE_CLASS_MAPPINGS	["integral2_0"] = cv2_integral2_0

class cv2_integral2_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sdepth"	: ("INT",),
				"sqdepth"	: ("INT",),
			},
			'optional': {
				"sum"	: ("NPARRAY",),
				"sqsum"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sdepth, sqdepth, sum=None, sqsum=None):
		ret = apply_function(cv2.integral2, [src, sum, sqsum, sdepth, sqdepth], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["integral2_1"] = "OpenCV integral2_1"
NODE_CLASS_MAPPINGS	["integral2_1"] = cv2_integral2_1

class cv2_integral3_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sdepth"	: ("INT",),
				"sqdepth"	: ("INT",),
			},
			'optional': {
				"sum"	: ("NPARRAY",),
				"sqsum"	: ("NPARRAY",),
				"tilted"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sdepth, sqdepth, sum=None, sqsum=None, tilted=None):
		ret = apply_function(cv2.integral3, [src, sum, sqsum, tilted, sdepth, sqdepth], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["integral3_0"] = "OpenCV integral3_0"
NODE_CLASS_MAPPINGS	["integral3_0"] = cv2_integral3_0

class cv2_integral3_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sdepth"	: ("INT",),
				"sqdepth"	: ("INT",),
			},
			'optional': {
				"sum"	: ("NPARRAY",),
				"sqsum"	: ("NPARRAY",),
				"tilted"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sdepth, sqdepth, sum=None, sqsum=None, tilted=None):
		ret = apply_function(cv2.integral3, [src, sum, sqsum, tilted, sdepth, sqdepth], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["integral3_1"] = "OpenCV integral3_1"
NODE_CLASS_MAPPINGS	["integral3_1"] = cv2_integral3_1

class cv2_intersectConvexConvex_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"p1"	: ("NPARRAY",),
				"p2"	: ("NPARRAY",),
				"handleNested"	: ("BOOLEAN",),
			},
			'optional': {
				"p12"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, p1, p2, handleNested, p12=None):
		ret = apply_function(cv2.intersectConvexConvex, [p1, p2, p12, handleNested], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["intersectConvexConvex_0"] = "OpenCV intersectConvexConvex_0"
NODE_CLASS_MAPPINGS	["intersectConvexConvex_0"] = cv2_intersectConvexConvex_0

class cv2_intersectConvexConvex_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"p1"	: ("NPARRAY",),
				"p2"	: ("NPARRAY",),
				"handleNested"	: ("BOOLEAN",),
			},
			'optional': {
				"p12"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, p1, p2, handleNested, p12=None):
		ret = apply_function(cv2.intersectConvexConvex, [p1, p2, p12, handleNested], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["intersectConvexConvex_1"] = "OpenCV intersectConvexConvex_1"
NODE_CLASS_MAPPINGS	["intersectConvexConvex_1"] = cv2_intersectConvexConvex_1

class cv2_invert_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, dst=None):
		ret = apply_function(cv2.invert, [src, dst, flags], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["invert_0"] = "OpenCV invert_0"
NODE_CLASS_MAPPINGS	["invert_0"] = cv2_invert_0

class cv2_invert_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, dst=None):
		ret = apply_function(cv2.invert, [src, dst, flags], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["invert_1"] = "OpenCV invert_1"
NODE_CLASS_MAPPINGS	["invert_1"] = cv2_invert_1

class cv2_invertAffineTransform_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"M"	: ("NPARRAY",),
			},
			'optional': {
				"iM"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, M, iM=None):
		ret = apply_function(cv2.invertAffineTransform, [M, iM], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["invertAffineTransform_0"] = "OpenCV invertAffineTransform_0"
NODE_CLASS_MAPPINGS	["invertAffineTransform_0"] = cv2_invertAffineTransform_0

class cv2_invertAffineTransform_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"M"	: ("NPARRAY",),
			},
			'optional': {
				"iM"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, M, iM=None):
		ret = apply_function(cv2.invertAffineTransform, [M, iM], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["invertAffineTransform_1"] = "OpenCV invertAffineTransform_1"
NODE_CLASS_MAPPINGS	["invertAffineTransform_1"] = cv2_invertAffineTransform_1

class cv2_isContourConvex_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"contour"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, contour):
		ret = apply_function(cv2.isContourConvex, [contour], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["isContourConvex_0"] = "OpenCV isContourConvex_0"
NODE_CLASS_MAPPINGS	["isContourConvex_0"] = cv2_isContourConvex_0

class cv2_isContourConvex_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"contour"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, contour):
		ret = apply_function(cv2.isContourConvex, [contour], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["isContourConvex_1"] = "OpenCV isContourConvex_1"
NODE_CLASS_MAPPINGS	["isContourConvex_1"] = cv2_isContourConvex_1

class cv2_kmeans_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"K"	: ("INT",),
				"bestLabels"	: ("NPARRAY",),
				"criteria"	: ("STRING",),
				"attempts"	: ("INT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"centers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, K, bestLabels, criteria, attempts, flags, centers=None):
		ret = apply_function(cv2.kmeans, [data, K, bestLabels, criteria, attempts, flags, centers], [0, 2, 6], [3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["kmeans_0"] = "OpenCV kmeans_0"
NODE_CLASS_MAPPINGS	["kmeans_0"] = cv2_kmeans_0

class cv2_kmeans_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"data"	: ("NPARRAY",),
				"K"	: ("INT",),
				"bestLabels"	: ("NPARRAY",),
				"criteria"	: ("STRING",),
				"attempts"	: ("INT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"centers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, data, K, bestLabels, criteria, attempts, flags, centers=None):
		ret = apply_function(cv2.kmeans, [data, K, bestLabels, criteria, attempts, flags, centers], [0, 2, 6], [3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["kmeans_1"] = "OpenCV kmeans_1"
NODE_CLASS_MAPPINGS	["kmeans_1"] = cv2_kmeans_1

class cv2_line_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"pt1"	: ("STRING",),
				"pt2"	: ("STRING",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, pt1, pt2, color, thickness, lineType, shift):
		ret = apply_function(cv2.line, [img, pt1, pt2, color, thickness, lineType, shift], [0], [1, 2, 3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["line_0"] = "OpenCV line_0"
NODE_CLASS_MAPPINGS	["line_0"] = cv2_line_0

class cv2_line_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"pt1"	: ("STRING",),
				"pt2"	: ("STRING",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, pt1, pt2, color, thickness, lineType, shift):
		ret = apply_function(cv2.line, [img, pt1, pt2, color, thickness, lineType, shift], [0], [1, 2, 3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["line_1"] = "OpenCV line_1"
NODE_CLASS_MAPPINGS	["line_1"] = cv2_line_1

class cv2_linearPolar_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"center"	: ("STRING",),
				"maxRadius"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, center, maxRadius, flags, dst=None):
		ret = apply_function(cv2.linearPolar, [src, center, maxRadius, flags, dst], [0, 4], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["linearPolar_0"] = "OpenCV linearPolar_0"
NODE_CLASS_MAPPINGS	["linearPolar_0"] = cv2_linearPolar_0

class cv2_linearPolar_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"center"	: ("STRING",),
				"maxRadius"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, center, maxRadius, flags, dst=None):
		ret = apply_function(cv2.linearPolar, [src, center, maxRadius, flags, dst], [0, 4], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["linearPolar_1"] = "OpenCV linearPolar_1"
NODE_CLASS_MAPPINGS	["linearPolar_1"] = cv2_linearPolar_1

class cv2_log_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.log, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["log_0"] = "OpenCV log_0"
NODE_CLASS_MAPPINGS	["log_0"] = cv2_log_0

class cv2_log_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.log, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["log_1"] = "OpenCV log_1"
NODE_CLASS_MAPPINGS	["log_1"] = cv2_log_1

class cv2_logPolar_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"center"	: ("STRING",),
				"M"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, center, M, flags, dst=None):
		ret = apply_function(cv2.logPolar, [src, center, M, flags, dst], [0, 4], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["logPolar_0"] = "OpenCV logPolar_0"
NODE_CLASS_MAPPINGS	["logPolar_0"] = cv2_logPolar_0

class cv2_logPolar_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"center"	: ("STRING",),
				"M"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, center, M, flags, dst=None):
		ret = apply_function(cv2.logPolar, [src, center, M, flags, dst], [0, 4], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["logPolar_1"] = "OpenCV logPolar_1"
NODE_CLASS_MAPPINGS	["logPolar_1"] = cv2_logPolar_1

class cv2_magnitude_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"x"	: ("NPARRAY",),
				"y"	: ("NPARRAY",),
			},
			'optional': {
				"magnitude"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, x, y, magnitude=None):
		ret = apply_function(cv2.magnitude, [x, y, magnitude], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["magnitude_0"] = "OpenCV magnitude_0"
NODE_CLASS_MAPPINGS	["magnitude_0"] = cv2_magnitude_0

class cv2_magnitude_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"x"	: ("NPARRAY",),
				"y"	: ("NPARRAY",),
			},
			'optional': {
				"magnitude"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, x, y, magnitude=None):
		ret = apply_function(cv2.magnitude, [x, y, magnitude], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["magnitude_1"] = "OpenCV magnitude_1"
NODE_CLASS_MAPPINGS	["magnitude_1"] = cv2_magnitude_1

class cv2_matMulDeriv_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"A"	: ("NPARRAY",),
				"B"	: ("NPARRAY",),
			},
			'optional': {
				"dABdA"	: ("NPARRAY",),
				"dABdB"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, A, B, dABdA=None, dABdB=None):
		ret = apply_function(cv2.matMulDeriv, [A, B, dABdA, dABdB], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["matMulDeriv_0"] = "OpenCV matMulDeriv_0"
NODE_CLASS_MAPPINGS	["matMulDeriv_0"] = cv2_matMulDeriv_0

class cv2_matMulDeriv_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"A"	: ("NPARRAY",),
				"B"	: ("NPARRAY",),
			},
			'optional': {
				"dABdA"	: ("NPARRAY",),
				"dABdB"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, A, B, dABdA=None, dABdB=None):
		ret = apply_function(cv2.matMulDeriv, [A, B, dABdA, dABdB], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["matMulDeriv_1"] = "OpenCV matMulDeriv_1"
NODE_CLASS_MAPPINGS	["matMulDeriv_1"] = cv2_matMulDeriv_1

class cv2_matchShapes_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"contour1"	: ("NPARRAY",),
				"contour2"	: ("NPARRAY",),
				"method"	: ("INT",),
				"parameter"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, contour1, contour2, method, parameter):
		ret = apply_function(cv2.matchShapes, [contour1, contour2, method, parameter], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["matchShapes_0"] = "OpenCV matchShapes_0"
NODE_CLASS_MAPPINGS	["matchShapes_0"] = cv2_matchShapes_0

class cv2_matchShapes_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"contour1"	: ("NPARRAY",),
				"contour2"	: ("NPARRAY",),
				"method"	: ("INT",),
				"parameter"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, contour1, contour2, method, parameter):
		ret = apply_function(cv2.matchShapes, [contour1, contour2, method, parameter], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["matchShapes_1"] = "OpenCV matchShapes_1"
NODE_CLASS_MAPPINGS	["matchShapes_1"] = cv2_matchShapes_1

class cv2_matchTemplate_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"templ"	: ("NPARRAY",),
				"method"	: ("INT",),
			},
			'optional': {
				"result"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, templ, method, result=None, mask=None):
		ret = apply_function(cv2.matchTemplate, [image, templ, method, result, mask], [0, 1, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["matchTemplate_0"] = "OpenCV matchTemplate_0"
NODE_CLASS_MAPPINGS	["matchTemplate_0"] = cv2_matchTemplate_0

class cv2_matchTemplate_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"templ"	: ("NPARRAY",),
				"method"	: ("INT",),
			},
			'optional': {
				"result"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, templ, method, result=None, mask=None):
		ret = apply_function(cv2.matchTemplate, [image, templ, method, result, mask], [0, 1, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["matchTemplate_1"] = "OpenCV matchTemplate_1"
NODE_CLASS_MAPPINGS	["matchTemplate_1"] = cv2_matchTemplate_1

class cv2_max_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None):
		ret = apply_function(cv2.max, [src1, src2, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["max_0"] = "OpenCV max_0"
NODE_CLASS_MAPPINGS	["max_0"] = cv2_max_0

class cv2_max_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None):
		ret = apply_function(cv2.max, [src1, src2, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["max_1"] = "OpenCV max_1"
NODE_CLASS_MAPPINGS	["max_1"] = cv2_max_1

class cv2_mean_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask=None):
		ret = apply_function(cv2.mean, [src, mask], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["mean_0"] = "OpenCV mean_0"
NODE_CLASS_MAPPINGS	["mean_0"] = cv2_mean_0

class cv2_mean_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask=None):
		ret = apply_function(cv2.mean, [src, mask], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["mean_1"] = "OpenCV mean_1"
NODE_CLASS_MAPPINGS	["mean_1"] = cv2_mean_1

class cv2_meanShift_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"probImage"	: ("NPARRAY",),
				"window"	: ("STRING",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT", "STRING",)
	RETURN_NAMES	= ("int", "literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, probImage, window, criteria):
		ret = apply_function(cv2.meanShift, [probImage, window, criteria], [0], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["meanShift_0"] = "OpenCV meanShift_0"
NODE_CLASS_MAPPINGS	["meanShift_0"] = cv2_meanShift_0

class cv2_meanShift_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"probImage"	: ("NPARRAY",),
				"window"	: ("STRING",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT", "STRING",)
	RETURN_NAMES	= ("int", "literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, probImage, window, criteria):
		ret = apply_function(cv2.meanShift, [probImage, window, criteria], [0], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["meanShift_1"] = "OpenCV meanShift_1"
NODE_CLASS_MAPPINGS	["meanShift_1"] = cv2_meanShift_1

class cv2_meanStdDev_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"mean"	: ("NPARRAY",),
				"stddev"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mean=None, stddev=None, mask=None):
		ret = apply_function(cv2.meanStdDev, [src, mean, stddev, mask], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["meanStdDev_0"] = "OpenCV meanStdDev_0"
NODE_CLASS_MAPPINGS	["meanStdDev_0"] = cv2_meanStdDev_0

class cv2_meanStdDev_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"mean"	: ("NPARRAY",),
				"stddev"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mean=None, stddev=None, mask=None):
		ret = apply_function(cv2.meanStdDev, [src, mean, stddev, mask], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["meanStdDev_1"] = "OpenCV meanStdDev_1"
NODE_CLASS_MAPPINGS	["meanStdDev_1"] = cv2_meanStdDev_1

class cv2_medianBlur_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, dst=None):
		ret = apply_function(cv2.medianBlur, [src, ksize, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["medianBlur_0"] = "OpenCV medianBlur_0"
NODE_CLASS_MAPPINGS	["medianBlur_0"] = cv2_medianBlur_0

class cv2_medianBlur_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, dst=None):
		ret = apply_function(cv2.medianBlur, [src, ksize, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["medianBlur_1"] = "OpenCV medianBlur_1"
NODE_CLASS_MAPPINGS	["medianBlur_1"] = cv2_medianBlur_1

class cv2_min_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None):
		ret = apply_function(cv2.min, [src1, src2, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["min_0"] = "OpenCV min_0"
NODE_CLASS_MAPPINGS	["min_0"] = cv2_min_0

class cv2_min_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dst=None):
		ret = apply_function(cv2.min, [src1, src2, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["min_1"] = "OpenCV min_1"
NODE_CLASS_MAPPINGS	["min_1"] = cv2_min_1

class cv2_minAreaRect_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points):
		ret = apply_function(cv2.minAreaRect, [points], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["minAreaRect_0"] = "OpenCV minAreaRect_0"
NODE_CLASS_MAPPINGS	["minAreaRect_0"] = cv2_minAreaRect_0

class cv2_minAreaRect_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points):
		ret = apply_function(cv2.minAreaRect, [points], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["minAreaRect_1"] = "OpenCV minAreaRect_1"
NODE_CLASS_MAPPINGS	["minAreaRect_1"] = cv2_minAreaRect_1

class cv2_minEnclosingCircle_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING", "FLOAT",)
	RETURN_NAMES	= ("literal", "float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points):
		ret = apply_function(cv2.minEnclosingCircle, [points], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["minEnclosingCircle_0"] = "OpenCV minEnclosingCircle_0"
NODE_CLASS_MAPPINGS	["minEnclosingCircle_0"] = cv2_minEnclosingCircle_0

class cv2_minEnclosingCircle_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING", "FLOAT",)
	RETURN_NAMES	= ("literal", "float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points):
		ret = apply_function(cv2.minEnclosingCircle, [points], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["minEnclosingCircle_1"] = "OpenCV minEnclosingCircle_1"
NODE_CLASS_MAPPINGS	["minEnclosingCircle_1"] = cv2_minEnclosingCircle_1

class cv2_minEnclosingTriangle_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				"triangle"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points, triangle=None):
		ret = apply_function(cv2.minEnclosingTriangle, [points, triangle], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["minEnclosingTriangle_0"] = "OpenCV minEnclosingTriangle_0"
NODE_CLASS_MAPPINGS	["minEnclosingTriangle_0"] = cv2_minEnclosingTriangle_0

class cv2_minEnclosingTriangle_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points"	: ("NPARRAY",),
			},
			'optional': {
				"triangle"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points, triangle=None):
		ret = apply_function(cv2.minEnclosingTriangle, [points, triangle], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["minEnclosingTriangle_1"] = "OpenCV minEnclosingTriangle_1"
NODE_CLASS_MAPPINGS	["minEnclosingTriangle_1"] = cv2_minEnclosingTriangle_1

class cv2_minMaxLoc_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "FLOAT", "STRING", "STRING",)
	RETURN_NAMES	= ("float_0", "float_1", "literal_2", "literal_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask=None):
		ret = apply_function(cv2.minMaxLoc, [src, mask], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["minMaxLoc_0"] = "OpenCV minMaxLoc_0"
NODE_CLASS_MAPPINGS	["minMaxLoc_0"] = cv2_minMaxLoc_0

class cv2_minMaxLoc_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "FLOAT", "STRING", "STRING",)
	RETURN_NAMES	= ("float_0", "float_1", "literal_2", "literal_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask=None):
		ret = apply_function(cv2.minMaxLoc, [src, mask], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["minMaxLoc_1"] = "OpenCV minMaxLoc_1"
NODE_CLASS_MAPPINGS	["minMaxLoc_1"] = cv2_minMaxLoc_1

class cv2_moments_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"array"	: ("NPARRAY",),
				"binaryImage"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, array, binaryImage):
		ret = apply_function(cv2.moments, [array, binaryImage], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["moments_0"] = "OpenCV moments_0"
NODE_CLASS_MAPPINGS	["moments_0"] = cv2_moments_0

class cv2_moments_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"array"	: ("NPARRAY",),
				"binaryImage"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, array, binaryImage):
		ret = apply_function(cv2.moments, [array, binaryImage], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["moments_1"] = "OpenCV moments_1"
NODE_CLASS_MAPPINGS	["moments_1"] = cv2_moments_1

class cv2_morphologyEx_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"op"	: ("INT",),
				"kernel"	: ("NPARRAY",),
				"anchor"	: ("STRING",),
				"iterations"	: ("INT",),
				"borderType"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, op, kernel, anchor, iterations, borderType, borderValue, dst=None):
		ret = apply_function(cv2.morphologyEx, [src, op, kernel, dst, anchor, iterations, borderType, borderValue], [0, 2, 3], [4, 7])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["morphologyEx_0"] = "OpenCV morphologyEx_0"
NODE_CLASS_MAPPINGS	["morphologyEx_0"] = cv2_morphologyEx_0

class cv2_morphologyEx_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"op"	: ("INT",),
				"kernel"	: ("NPARRAY",),
				"anchor"	: ("STRING",),
				"iterations"	: ("INT",),
				"borderType"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, op, kernel, anchor, iterations, borderType, borderValue, dst=None):
		ret = apply_function(cv2.morphologyEx, [src, op, kernel, dst, anchor, iterations, borderType, borderValue], [0, 2, 3], [4, 7])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["morphologyEx_1"] = "OpenCV morphologyEx_1"
NODE_CLASS_MAPPINGS	["morphologyEx_1"] = cv2_morphologyEx_1

class cv2_moveWindow_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"x"	: ("INT",),
				"y"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, x, y):
		ret = apply_function(cv2.moveWindow, [winname, x, y], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["moveWindow_0"] = "OpenCV moveWindow_0"
NODE_CLASS_MAPPINGS	["moveWindow_0"] = cv2_moveWindow_0

class cv2_mulSpectrums_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"a"	: ("NPARRAY",),
				"b"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"conjB"	: ("BOOLEAN",),
			},
			'optional': {
				"c"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, a, b, flags, conjB, c=None):
		ret = apply_function(cv2.mulSpectrums, [a, b, flags, c, conjB], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["mulSpectrums_0"] = "OpenCV mulSpectrums_0"
NODE_CLASS_MAPPINGS	["mulSpectrums_0"] = cv2_mulSpectrums_0

class cv2_mulSpectrums_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"a"	: ("NPARRAY",),
				"b"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"conjB"	: ("BOOLEAN",),
			},
			'optional': {
				"c"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, a, b, flags, conjB, c=None):
		ret = apply_function(cv2.mulSpectrums, [a, b, flags, c, conjB], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["mulSpectrums_1"] = "OpenCV mulSpectrums_1"
NODE_CLASS_MAPPINGS	["mulSpectrums_1"] = cv2_mulSpectrums_1

class cv2_mulTransposed_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"aTa"	: ("BOOLEAN",),
				"scale"	: ("FLOAT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"delta"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, aTa, scale, dtype, dst=None, delta=None):
		ret = apply_function(cv2.mulTransposed, [src, aTa, dst, delta, scale, dtype], [0, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["mulTransposed_0"] = "OpenCV mulTransposed_0"
NODE_CLASS_MAPPINGS	["mulTransposed_0"] = cv2_mulTransposed_0

class cv2_mulTransposed_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"aTa"	: ("BOOLEAN",),
				"scale"	: ("FLOAT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"delta"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, aTa, scale, dtype, dst=None, delta=None):
		ret = apply_function(cv2.mulTransposed, [src, aTa, dst, delta, scale, dtype], [0, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["mulTransposed_1"] = "OpenCV mulTransposed_1"
NODE_CLASS_MAPPINGS	["mulTransposed_1"] = cv2_mulTransposed_1

class cv2_multiply_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"scale"	: ("FLOAT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, scale, dtype, dst=None):
		ret = apply_function(cv2.multiply, [src1, src2, dst, scale, dtype], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["multiply_0"] = "OpenCV multiply_0"
NODE_CLASS_MAPPINGS	["multiply_0"] = cv2_multiply_0

class cv2_multiply_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"scale"	: ("FLOAT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, scale, dtype, dst=None):
		ret = apply_function(cv2.multiply, [src1, src2, dst, scale, dtype], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["multiply_1"] = "OpenCV multiply_1"
NODE_CLASS_MAPPINGS	["multiply_1"] = cv2_multiply_1

class cv2_namedWindow_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, flags):
		ret = apply_function(cv2.namedWindow, [winname, flags], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["namedWindow_0"] = "OpenCV namedWindow_0"
NODE_CLASS_MAPPINGS	["namedWindow_0"] = cv2_namedWindow_0

class cv2_norm_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"normType"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, normType, mask=None):
		ret = apply_function(cv2.norm, [src1, normType, mask], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["norm_0"] = "OpenCV norm_0"
NODE_CLASS_MAPPINGS	["norm_0"] = cv2_norm_0

class cv2_norm_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"normType"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, normType, mask=None):
		ret = apply_function(cv2.norm, [src1, normType, mask], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["norm_1"] = "OpenCV norm_1"
NODE_CLASS_MAPPINGS	["norm_1"] = cv2_norm_1

class cv2_norm_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"normType"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, normType, mask=None):
		ret = apply_function(cv2.norm, [src1, src2, normType, mask], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["norm_2"] = "OpenCV norm_2"
NODE_CLASS_MAPPINGS	["norm_2"] = cv2_norm_2

class cv2_norm_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"normType"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, normType, mask=None):
		ret = apply_function(cv2.norm, [src1, src2, normType, mask], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["norm_3"] = "OpenCV norm_3"
NODE_CLASS_MAPPINGS	["norm_3"] = cv2_norm_3

class cv2_normalize_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"beta"	: ("FLOAT",),
				"norm_type"	: ("INT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, alpha, beta, norm_type, dtype, mask=None):
		ret = apply_function(cv2.normalize, [src, dst, alpha, beta, norm_type, dtype, mask], [0, 1, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["normalize_0"] = "OpenCV normalize_0"
NODE_CLASS_MAPPINGS	["normalize_0"] = cv2_normalize_0

class cv2_normalize_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"beta"	: ("FLOAT",),
				"norm_type"	: ("INT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, alpha, beta, norm_type, dtype, mask=None):
		ret = apply_function(cv2.normalize, [src, dst, alpha, beta, norm_type, dtype, mask], [0, 1, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["normalize_1"] = "OpenCV normalize_1"
NODE_CLASS_MAPPINGS	["normalize_1"] = cv2_normalize_1

class cv2_patchNaNs_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"a"	: ("NPARRAY",),
				"val"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, a, val):
		ret = apply_function(cv2.patchNaNs, [a, val], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["patchNaNs_0"] = "OpenCV patchNaNs_0"
NODE_CLASS_MAPPINGS	["patchNaNs_0"] = cv2_patchNaNs_0

class cv2_patchNaNs_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"a"	: ("NPARRAY",),
				"val"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, a, val):
		ret = apply_function(cv2.patchNaNs, [a, val], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["patchNaNs_1"] = "OpenCV patchNaNs_1"
NODE_CLASS_MAPPINGS	["patchNaNs_1"] = cv2_patchNaNs_1

class cv2_pencilSketch_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sigma_s"	: ("FLOAT",),
				"sigma_r"	: ("FLOAT",),
				"shade_factor"	: ("FLOAT",),
			},
			'optional': {
				"dst1"	: ("NPARRAY",),
				"dst2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sigma_s, sigma_r, shade_factor, dst1=None, dst2=None):
		ret = apply_function(cv2.pencilSketch, [src, dst1, dst2, sigma_s, sigma_r, shade_factor], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pencilSketch_0"] = "OpenCV pencilSketch_0"
NODE_CLASS_MAPPINGS	["pencilSketch_0"] = cv2_pencilSketch_0

class cv2_pencilSketch_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sigma_s"	: ("FLOAT",),
				"sigma_r"	: ("FLOAT",),
				"shade_factor"	: ("FLOAT",),
			},
			'optional': {
				"dst1"	: ("NPARRAY",),
				"dst2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sigma_s, sigma_r, shade_factor, dst1=None, dst2=None):
		ret = apply_function(cv2.pencilSketch, [src, dst1, dst2, sigma_s, sigma_r, shade_factor], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pencilSketch_1"] = "OpenCV pencilSketch_1"
NODE_CLASS_MAPPINGS	["pencilSketch_1"] = cv2_pencilSketch_1

class cv2_perspectiveTransform_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"m"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, m, dst=None):
		ret = apply_function(cv2.perspectiveTransform, [src, m, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["perspectiveTransform_0"] = "OpenCV perspectiveTransform_0"
NODE_CLASS_MAPPINGS	["perspectiveTransform_0"] = cv2_perspectiveTransform_0

class cv2_perspectiveTransform_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"m"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, m, dst=None):
		ret = apply_function(cv2.perspectiveTransform, [src, m, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["perspectiveTransform_1"] = "OpenCV perspectiveTransform_1"
NODE_CLASS_MAPPINGS	["perspectiveTransform_1"] = cv2_perspectiveTransform_1

class cv2_phase_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"x"	: ("NPARRAY",),
				"y"	: ("NPARRAY",),
				"angleInDegrees"	: ("BOOLEAN",),
			},
			'optional': {
				"angle"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, x, y, angleInDegrees, angle=None):
		ret = apply_function(cv2.phase, [x, y, angle, angleInDegrees], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["phase_0"] = "OpenCV phase_0"
NODE_CLASS_MAPPINGS	["phase_0"] = cv2_phase_0

class cv2_phase_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"x"	: ("NPARRAY",),
				"y"	: ("NPARRAY",),
				"angleInDegrees"	: ("BOOLEAN",),
			},
			'optional': {
				"angle"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, x, y, angleInDegrees, angle=None):
		ret = apply_function(cv2.phase, [x, y, angle, angleInDegrees], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["phase_1"] = "OpenCV phase_1"
NODE_CLASS_MAPPINGS	["phase_1"] = cv2_phase_1

class cv2_phaseCorrelate_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"window"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("STRING", "FLOAT",)
	RETURN_NAMES	= ("literal", "float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, window=None):
		ret = apply_function(cv2.phaseCorrelate, [src1, src2, window], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["phaseCorrelate_0"] = "OpenCV phaseCorrelate_0"
NODE_CLASS_MAPPINGS	["phaseCorrelate_0"] = cv2_phaseCorrelate_0

class cv2_phaseCorrelate_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"window"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("STRING", "FLOAT",)
	RETURN_NAMES	= ("literal", "float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, window=None):
		ret = apply_function(cv2.phaseCorrelate, [src1, src2, window], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["phaseCorrelate_1"] = "OpenCV phaseCorrelate_1"
NODE_CLASS_MAPPINGS	["phaseCorrelate_1"] = cv2_phaseCorrelate_1

class cv2_pointPolygonTest_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"contour"	: ("NPARRAY",),
				"pt"	: ("STRING",),
				"measureDist"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, contour, pt, measureDist):
		ret = apply_function(cv2.pointPolygonTest, [contour, pt, measureDist], [0], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pointPolygonTest_0"] = "OpenCV pointPolygonTest_0"
NODE_CLASS_MAPPINGS	["pointPolygonTest_0"] = cv2_pointPolygonTest_0

class cv2_pointPolygonTest_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"contour"	: ("NPARRAY",),
				"pt"	: ("STRING",),
				"measureDist"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, contour, pt, measureDist):
		ret = apply_function(cv2.pointPolygonTest, [contour, pt, measureDist], [0], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pointPolygonTest_1"] = "OpenCV pointPolygonTest_1"
NODE_CLASS_MAPPINGS	["pointPolygonTest_1"] = cv2_pointPolygonTest_1

class cv2_polarToCart_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"magnitude"	: ("NPARRAY",),
				"angle"	: ("NPARRAY",),
				"angleInDegrees"	: ("BOOLEAN",),
			},
			'optional': {
				"x"	: ("NPARRAY",),
				"y"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, magnitude, angle, angleInDegrees, x=None, y=None):
		ret = apply_function(cv2.polarToCart, [magnitude, angle, x, y, angleInDegrees], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["polarToCart_0"] = "OpenCV polarToCart_0"
NODE_CLASS_MAPPINGS	["polarToCart_0"] = cv2_polarToCart_0

class cv2_polarToCart_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"magnitude"	: ("NPARRAY",),
				"angle"	: ("NPARRAY",),
				"angleInDegrees"	: ("BOOLEAN",),
			},
			'optional': {
				"x"	: ("NPARRAY",),
				"y"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, magnitude, angle, angleInDegrees, x=None, y=None):
		ret = apply_function(cv2.polarToCart, [magnitude, angle, x, y, angleInDegrees], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["polarToCart_1"] = "OpenCV polarToCart_1"
NODE_CLASS_MAPPINGS	["polarToCart_1"] = cv2_polarToCart_1

class cv2_pollKey_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.pollKey, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pollKey_0"] = "OpenCV pollKey_0"
NODE_CLASS_MAPPINGS	["pollKey_0"] = cv2_pollKey_0

class cv2_pow_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"power"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, power, dst=None):
		ret = apply_function(cv2.pow, [src, power, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pow_0"] = "OpenCV pow_0"
NODE_CLASS_MAPPINGS	["pow_0"] = cv2_pow_0

class cv2_pow_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"power"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, power, dst=None):
		ret = apply_function(cv2.pow, [src, power, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pow_1"] = "OpenCV pow_1"
NODE_CLASS_MAPPINGS	["pow_1"] = cv2_pow_1

class cv2_preCornerDetect_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("INT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, borderType, dst=None):
		ret = apply_function(cv2.preCornerDetect, [src, ksize, dst, borderType], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["preCornerDetect_0"] = "OpenCV preCornerDetect_0"
NODE_CLASS_MAPPINGS	["preCornerDetect_0"] = cv2_preCornerDetect_0

class cv2_preCornerDetect_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("INT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, borderType, dst=None):
		ret = apply_function(cv2.preCornerDetect, [src, ksize, dst, borderType], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["preCornerDetect_1"] = "OpenCV preCornerDetect_1"
NODE_CLASS_MAPPINGS	["preCornerDetect_1"] = cv2_preCornerDetect_1

class cv2_projectPoints_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"objectPoints"	: ("NPARRAY",),
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"aspectRatio"	: ("FLOAT",),
			},
			'optional': {
				"imagePoints"	: ("NPARRAY",),
				"jacobian"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, objectPoints, rvec, tvec, cameraMatrix, distCoeffs, aspectRatio, imagePoints=None, jacobian=None):
		ret = apply_function(cv2.projectPoints, [objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints, jacobian, aspectRatio], [0, 1, 2, 3, 4, 5, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["projectPoints_0"] = "OpenCV projectPoints_0"
NODE_CLASS_MAPPINGS	["projectPoints_0"] = cv2_projectPoints_0

class cv2_projectPoints_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"objectPoints"	: ("NPARRAY",),
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"aspectRatio"	: ("FLOAT",),
			},
			'optional': {
				"imagePoints"	: ("NPARRAY",),
				"jacobian"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, objectPoints, rvec, tvec, cameraMatrix, distCoeffs, aspectRatio, imagePoints=None, jacobian=None):
		ret = apply_function(cv2.projectPoints, [objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints, jacobian, aspectRatio], [0, 1, 2, 3, 4, 5, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["projectPoints_1"] = "OpenCV projectPoints_1"
NODE_CLASS_MAPPINGS	["projectPoints_1"] = cv2_projectPoints_1

class cv2_putText_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"text"	: ("STRING",),
				"org"	: ("STRING",),
				"fontFace"	: ("INT",),
				"fontScale"	: ("FLOAT",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"bottomLeftOrigin"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin):
		ret = apply_function(cv2.putText, [img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin], [0], [2, 5])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["putText_0"] = "OpenCV putText_0"
NODE_CLASS_MAPPINGS	["putText_0"] = cv2_putText_0

class cv2_putText_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"text"	: ("STRING",),
				"org"	: ("STRING",),
				"fontFace"	: ("INT",),
				"fontScale"	: ("FLOAT",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"bottomLeftOrigin"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin):
		ret = apply_function(cv2.putText, [img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin], [0], [2, 5])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["putText_1"] = "OpenCV putText_1"
NODE_CLASS_MAPPINGS	["putText_1"] = cv2_putText_1

class cv2_pyrDown_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dstsize"	: ("STRING",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dstsize, borderType, dst=None):
		ret = apply_function(cv2.pyrDown, [src, dst, dstsize, borderType], [0, 1], [2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pyrDown_0"] = "OpenCV pyrDown_0"
NODE_CLASS_MAPPINGS	["pyrDown_0"] = cv2_pyrDown_0

class cv2_pyrDown_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dstsize"	: ("STRING",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dstsize, borderType, dst=None):
		ret = apply_function(cv2.pyrDown, [src, dst, dstsize, borderType], [0, 1], [2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pyrDown_1"] = "OpenCV pyrDown_1"
NODE_CLASS_MAPPINGS	["pyrDown_1"] = cv2_pyrDown_1

class cv2_pyrMeanShiftFiltering_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sp"	: ("FLOAT",),
				"sr"	: ("FLOAT",),
				"maxLevel"	: ("INT",),
				"termcrit"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sp, sr, maxLevel, termcrit, dst=None):
		ret = apply_function(cv2.pyrMeanShiftFiltering, [src, sp, sr, dst, maxLevel, termcrit], [0, 3], [5])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pyrMeanShiftFiltering_0"] = "OpenCV pyrMeanShiftFiltering_0"
NODE_CLASS_MAPPINGS	["pyrMeanShiftFiltering_0"] = cv2_pyrMeanShiftFiltering_0

class cv2_pyrMeanShiftFiltering_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sp"	: ("FLOAT",),
				"sr"	: ("FLOAT",),
				"maxLevel"	: ("INT",),
				"termcrit"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sp, sr, maxLevel, termcrit, dst=None):
		ret = apply_function(cv2.pyrMeanShiftFiltering, [src, sp, sr, dst, maxLevel, termcrit], [0, 3], [5])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pyrMeanShiftFiltering_1"] = "OpenCV pyrMeanShiftFiltering_1"
NODE_CLASS_MAPPINGS	["pyrMeanShiftFiltering_1"] = cv2_pyrMeanShiftFiltering_1

class cv2_pyrUp_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dstsize"	: ("STRING",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dstsize, borderType, dst=None):
		ret = apply_function(cv2.pyrUp, [src, dst, dstsize, borderType], [0, 1], [2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pyrUp_0"] = "OpenCV pyrUp_0"
NODE_CLASS_MAPPINGS	["pyrUp_0"] = cv2_pyrUp_0

class cv2_pyrUp_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dstsize"	: ("STRING",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dstsize, borderType, dst=None):
		ret = apply_function(cv2.pyrUp, [src, dst, dstsize, borderType], [0, 1], [2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["pyrUp_1"] = "OpenCV pyrUp_1"
NODE_CLASS_MAPPINGS	["pyrUp_1"] = cv2_pyrUp_1

class cv2_randShuffle_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"dst"	: ("NPARRAY",),
				"iterFactor"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, dst, iterFactor):
		ret = apply_function(cv2.randShuffle, [dst, iterFactor], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["randShuffle_0"] = "OpenCV randShuffle_0"
NODE_CLASS_MAPPINGS	["randShuffle_0"] = cv2_randShuffle_0

class cv2_randShuffle_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"dst"	: ("NPARRAY",),
				"iterFactor"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, dst, iterFactor):
		ret = apply_function(cv2.randShuffle, [dst, iterFactor], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["randShuffle_1"] = "OpenCV randShuffle_1"
NODE_CLASS_MAPPINGS	["randShuffle_1"] = cv2_randShuffle_1

class cv2_randn_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"dst"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"stddev"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, dst, mean, stddev):
		ret = apply_function(cv2.randn, [dst, mean, stddev], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["randn_0"] = "OpenCV randn_0"
NODE_CLASS_MAPPINGS	["randn_0"] = cv2_randn_0

class cv2_randn_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"dst"	: ("NPARRAY",),
				"mean"	: ("NPARRAY",),
				"stddev"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, dst, mean, stddev):
		ret = apply_function(cv2.randn, [dst, mean, stddev], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["randn_1"] = "OpenCV randn_1"
NODE_CLASS_MAPPINGS	["randn_1"] = cv2_randn_1

class cv2_randu_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"dst"	: ("NPARRAY",),
				"low"	: ("NPARRAY",),
				"high"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, dst, low, high):
		ret = apply_function(cv2.randu, [dst, low, high], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["randu_0"] = "OpenCV randu_0"
NODE_CLASS_MAPPINGS	["randu_0"] = cv2_randu_0

class cv2_randu_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"dst"	: ("NPARRAY",),
				"low"	: ("NPARRAY",),
				"high"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, dst, low, high):
		ret = apply_function(cv2.randu, [dst, low, high], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["randu_1"] = "OpenCV randu_1"
NODE_CLASS_MAPPINGS	["randu_1"] = cv2_randu_1

class cv2_readOpticalFlow_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"path"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, path):
		ret = apply_function(cv2.readOpticalFlow, [path], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["readOpticalFlow_0"] = "OpenCV readOpticalFlow_0"
NODE_CLASS_MAPPINGS	["readOpticalFlow_0"] = cv2_readOpticalFlow_0

class cv2_recoverPose_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"cameraMatrix1"	: ("NPARRAY",),
				"distCoeffs1"	: ("NPARRAY",),
				"cameraMatrix2"	: ("NPARRAY",),
				"distCoeffs2"	: ("NPARRAY",),
				"method"	: ("INT",),
				"prob"	: ("FLOAT",),
				"threshold"	: ("FLOAT",),
			},
			'optional': {
				"E"	: ("NPARRAY",),
				"R"	: ("NPARRAY",),
				"t"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3", "nparray_4",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, method, prob, threshold, E=None, R=None, t=None, mask=None):
		ret = apply_function(cv2.recoverPose, [points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, E, R, t, method, prob, threshold, mask], [0, 1, 2, 3, 4, 5, 6, 7, 8, 12], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["recoverPose_0"] = "OpenCV recoverPose_0"
NODE_CLASS_MAPPINGS	["recoverPose_0"] = cv2_recoverPose_0

class cv2_recoverPose_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"cameraMatrix1"	: ("NPARRAY",),
				"distCoeffs1"	: ("NPARRAY",),
				"cameraMatrix2"	: ("NPARRAY",),
				"distCoeffs2"	: ("NPARRAY",),
				"method"	: ("INT",),
				"prob"	: ("FLOAT",),
				"threshold"	: ("FLOAT",),
			},
			'optional': {
				"E"	: ("NPARRAY",),
				"R"	: ("NPARRAY",),
				"t"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3", "nparray_4",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, method, prob, threshold, E=None, R=None, t=None, mask=None):
		ret = apply_function(cv2.recoverPose, [points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, E, R, t, method, prob, threshold, mask], [0, 1, 2, 3, 4, 5, 6, 7, 8, 12], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["recoverPose_1"] = "OpenCV recoverPose_1"
NODE_CLASS_MAPPINGS	["recoverPose_1"] = cv2_recoverPose_1

class cv2_recoverPose_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"E"	: ("NPARRAY",),
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
			},
			'optional': {
				"R"	: ("NPARRAY",),
				"t"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, E, points1, points2, cameraMatrix, R=None, t=None, mask=None):
		ret = apply_function(cv2.recoverPose, [E, points1, points2, cameraMatrix, R, t, mask], [0, 1, 2, 3, 4, 5, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["recoverPose_2"] = "OpenCV recoverPose_2"
NODE_CLASS_MAPPINGS	["recoverPose_2"] = cv2_recoverPose_2

class cv2_recoverPose_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"E"	: ("NPARRAY",),
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
			},
			'optional': {
				"R"	: ("NPARRAY",),
				"t"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, E, points1, points2, cameraMatrix, R=None, t=None, mask=None):
		ret = apply_function(cv2.recoverPose, [E, points1, points2, cameraMatrix, R, t, mask], [0, 1, 2, 3, 4, 5, 6], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["recoverPose_3"] = "OpenCV recoverPose_3"
NODE_CLASS_MAPPINGS	["recoverPose_3"] = cv2_recoverPose_3

class cv2_recoverPose_4:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"E"	: ("NPARRAY",),
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"focal"	: ("FLOAT",),
				"pp"	: ("STRING",),
			},
			'optional': {
				"R"	: ("NPARRAY",),
				"t"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, E, points1, points2, focal, pp, R=None, t=None, mask=None):
		ret = apply_function(cv2.recoverPose, [E, points1, points2, R, t, focal, pp, mask], [0, 1, 2, 3, 4, 7], [6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["recoverPose_4"] = "OpenCV recoverPose_4"
NODE_CLASS_MAPPINGS	["recoverPose_4"] = cv2_recoverPose_4

class cv2_recoverPose_5:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"E"	: ("NPARRAY",),
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"focal"	: ("FLOAT",),
				"pp"	: ("STRING",),
			},
			'optional': {
				"R"	: ("NPARRAY",),
				"t"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, E, points1, points2, focal, pp, R=None, t=None, mask=None):
		ret = apply_function(cv2.recoverPose, [E, points1, points2, R, t, focal, pp, mask], [0, 1, 2, 3, 4, 7], [6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["recoverPose_5"] = "OpenCV recoverPose_5"
NODE_CLASS_MAPPINGS	["recoverPose_5"] = cv2_recoverPose_5

class cv2_recoverPose_6:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"E"	: ("NPARRAY",),
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distanceThresh"	: ("FLOAT",),
			},
			'optional': {
				"R"	: ("NPARRAY",),
				"t"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"triangulatedPoints"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3", "nparray_4",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, E, points1, points2, cameraMatrix, distanceThresh, R=None, t=None, mask=None, triangulatedPoints=None):
		ret = apply_function(cv2.recoverPose, [E, points1, points2, cameraMatrix, distanceThresh, R, t, mask, triangulatedPoints], [0, 1, 2, 3, 5, 6, 7, 8], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["recoverPose_6"] = "OpenCV recoverPose_6"
NODE_CLASS_MAPPINGS	["recoverPose_6"] = cv2_recoverPose_6

class cv2_recoverPose_7:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"E"	: ("NPARRAY",),
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distanceThresh"	: ("FLOAT",),
			},
			'optional': {
				"R"	: ("NPARRAY",),
				"t"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"triangulatedPoints"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray_1", "nparray_2", "nparray_3", "nparray_4",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, E, points1, points2, cameraMatrix, distanceThresh, R=None, t=None, mask=None, triangulatedPoints=None):
		ret = apply_function(cv2.recoverPose, [E, points1, points2, cameraMatrix, distanceThresh, R, t, mask, triangulatedPoints], [0, 1, 2, 3, 5, 6, 7, 8], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["recoverPose_7"] = "OpenCV recoverPose_7"
NODE_CLASS_MAPPINGS	["recoverPose_7"] = cv2_recoverPose_7

class cv2_rectangle_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"pt1"	: ("STRING",),
				"pt2"	: ("STRING",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, pt1, pt2, color, thickness, lineType, shift):
		ret = apply_function(cv2.rectangle, [img, pt1, pt2, color, thickness, lineType, shift], [0], [1, 2, 3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["rectangle_0"] = "OpenCV rectangle_0"
NODE_CLASS_MAPPINGS	["rectangle_0"] = cv2_rectangle_0

class cv2_rectangle_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"pt1"	: ("STRING",),
				"pt2"	: ("STRING",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, pt1, pt2, color, thickness, lineType, shift):
		ret = apply_function(cv2.rectangle, [img, pt1, pt2, color, thickness, lineType, shift], [0], [1, 2, 3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["rectangle_1"] = "OpenCV rectangle_1"
NODE_CLASS_MAPPINGS	["rectangle_1"] = cv2_rectangle_1

class cv2_rectangle_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"rec"	: ("STRING",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, rec, color, thickness, lineType, shift):
		ret = apply_function(cv2.rectangle, [img, rec, color, thickness, lineType, shift], [0], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["rectangle_2"] = "OpenCV rectangle_2"
NODE_CLASS_MAPPINGS	["rectangle_2"] = cv2_rectangle_2

class cv2_rectangle_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"rec"	: ("STRING",),
				"color"	: ("STRING",),
				"thickness"	: ("INT",),
				"lineType"	: ("INT",),
				"shift"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, rec, color, thickness, lineType, shift):
		ret = apply_function(cv2.rectangle, [img, rec, color, thickness, lineType, shift], [0], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["rectangle_3"] = "OpenCV rectangle_3"
NODE_CLASS_MAPPINGS	["rectangle_3"] = cv2_rectangle_3

class cv2_rectangleIntersectionArea_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"a"	: ("STRING",),
				"b"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, a, b):
		ret = apply_function(cv2.rectangleIntersectionArea, [a, b], [], [0, 1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["rectangleIntersectionArea_0"] = "OpenCV rectangleIntersectionArea_0"
NODE_CLASS_MAPPINGS	["rectangleIntersectionArea_0"] = cv2_rectangleIntersectionArea_0

class cv2_reduce_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dim"	: ("INT",),
				"rtype"	: ("INT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dim, rtype, dtype, dst=None):
		ret = apply_function(cv2.reduce, [src, dim, rtype, dst, dtype], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["reduce_0"] = "OpenCV reduce_0"
NODE_CLASS_MAPPINGS	["reduce_0"] = cv2_reduce_0

class cv2_reduce_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dim"	: ("INT",),
				"rtype"	: ("INT",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dim, rtype, dtype, dst=None):
		ret = apply_function(cv2.reduce, [src, dim, rtype, dst, dtype], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["reduce_1"] = "OpenCV reduce_1"
NODE_CLASS_MAPPINGS	["reduce_1"] = cv2_reduce_1

class cv2_reduceArgMax_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"axis"	: ("INT",),
				"lastIndex"	: ("BOOLEAN",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, axis, lastIndex, dst=None):
		ret = apply_function(cv2.reduceArgMax, [src, axis, dst, lastIndex], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["reduceArgMax_0"] = "OpenCV reduceArgMax_0"
NODE_CLASS_MAPPINGS	["reduceArgMax_0"] = cv2_reduceArgMax_0

class cv2_reduceArgMax_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"axis"	: ("INT",),
				"lastIndex"	: ("BOOLEAN",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, axis, lastIndex, dst=None):
		ret = apply_function(cv2.reduceArgMax, [src, axis, dst, lastIndex], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["reduceArgMax_1"] = "OpenCV reduceArgMax_1"
NODE_CLASS_MAPPINGS	["reduceArgMax_1"] = cv2_reduceArgMax_1

class cv2_reduceArgMin_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"axis"	: ("INT",),
				"lastIndex"	: ("BOOLEAN",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, axis, lastIndex, dst=None):
		ret = apply_function(cv2.reduceArgMin, [src, axis, dst, lastIndex], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["reduceArgMin_0"] = "OpenCV reduceArgMin_0"
NODE_CLASS_MAPPINGS	["reduceArgMin_0"] = cv2_reduceArgMin_0

class cv2_reduceArgMin_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"axis"	: ("INT",),
				"lastIndex"	: ("BOOLEAN",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, axis, lastIndex, dst=None):
		ret = apply_function(cv2.reduceArgMin, [src, axis, dst, lastIndex], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["reduceArgMin_1"] = "OpenCV reduceArgMin_1"
NODE_CLASS_MAPPINGS	["reduceArgMin_1"] = cv2_reduceArgMin_1

class cv2_remap_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"map1"	: ("NPARRAY",),
				"map2"	: ("NPARRAY",),
				"interpolation"	: ("INT",),
				"borderMode"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, map1, map2, interpolation, borderMode, borderValue, dst=None):
		ret = apply_function(cv2.remap, [src, map1, map2, interpolation, dst, borderMode, borderValue], [0, 1, 2, 4], [6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["remap_0"] = "OpenCV remap_0"
NODE_CLASS_MAPPINGS	["remap_0"] = cv2_remap_0

class cv2_remap_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"map1"	: ("NPARRAY",),
				"map2"	: ("NPARRAY",),
				"interpolation"	: ("INT",),
				"borderMode"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, map1, map2, interpolation, borderMode, borderValue, dst=None):
		ret = apply_function(cv2.remap, [src, map1, map2, interpolation, dst, borderMode, borderValue], [0, 1, 2, 4], [6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["remap_1"] = "OpenCV remap_1"
NODE_CLASS_MAPPINGS	["remap_1"] = cv2_remap_1

class cv2_repeat_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ny"	: ("INT",),
				"nx"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ny, nx, dst=None):
		ret = apply_function(cv2.repeat, [src, ny, nx, dst], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["repeat_0"] = "OpenCV repeat_0"
NODE_CLASS_MAPPINGS	["repeat_0"] = cv2_repeat_0

class cv2_repeat_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ny"	: ("INT",),
				"nx"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ny, nx, dst=None):
		ret = apply_function(cv2.repeat, [src, ny, nx, dst], [0, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["repeat_1"] = "OpenCV repeat_1"
NODE_CLASS_MAPPINGS	["repeat_1"] = cv2_repeat_1

class cv2_reprojectImageTo3D_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"disparity"	: ("NPARRAY",),
				"Q"	: ("NPARRAY",),
				"handleMissingValues"	: ("BOOLEAN",),
				"ddepth"	: ("INT",),
			},
			'optional': {
				"_3dImage"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, disparity, Q, handleMissingValues, ddepth, _3dImage=None):
		ret = apply_function(cv2.reprojectImageTo3D, [disparity, Q, _3dImage, handleMissingValues, ddepth], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["reprojectImageTo3D_0"] = "OpenCV reprojectImageTo3D_0"
NODE_CLASS_MAPPINGS	["reprojectImageTo3D_0"] = cv2_reprojectImageTo3D_0

class cv2_reprojectImageTo3D_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"disparity"	: ("NPARRAY",),
				"Q"	: ("NPARRAY",),
				"handleMissingValues"	: ("BOOLEAN",),
				"ddepth"	: ("INT",),
			},
			'optional': {
				"_3dImage"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, disparity, Q, handleMissingValues, ddepth, _3dImage=None):
		ret = apply_function(cv2.reprojectImageTo3D, [disparity, Q, _3dImage, handleMissingValues, ddepth], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["reprojectImageTo3D_1"] = "OpenCV reprojectImageTo3D_1"
NODE_CLASS_MAPPINGS	["reprojectImageTo3D_1"] = cv2_reprojectImageTo3D_1

class cv2_resize_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"fx"	: ("FLOAT",),
				"fy"	: ("FLOAT",),
				"interpolation"	: ("INT",),
			},
			'optional': {
				"dsize"	: ("STRING",),
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, fx, fy, interpolation, dsize=None, dst=None):
		ret = apply_function(cv2.resize, [src, dsize, dst, fx, fy, interpolation], [0, 2], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["resize_0"] = "OpenCV resize_0"
NODE_CLASS_MAPPINGS	["resize_0"] = cv2_resize_0

class cv2_resize_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"fx"	: ("FLOAT",),
				"fy"	: ("FLOAT",),
				"interpolation"	: ("INT",),
			},
			'optional': {
				"dsize"	: ("STRING",),
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, fx, fy, interpolation, dsize=None, dst=None):
		ret = apply_function(cv2.resize, [src, dsize, dst, fx, fy, interpolation], [0, 2], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["resize_1"] = "OpenCV resize_1"
NODE_CLASS_MAPPINGS	["resize_1"] = cv2_resize_1

class cv2_resizeWindow_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"width"	: ("INT",),
				"height"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, width, height):
		ret = apply_function(cv2.resizeWindow, [winname, width, height], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["resizeWindow_0"] = "OpenCV resizeWindow_0"
NODE_CLASS_MAPPINGS	["resizeWindow_0"] = cv2_resizeWindow_0

class cv2_resizeWindow_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"size"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, size):
		ret = apply_function(cv2.resizeWindow, [winname, size], [], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["resizeWindow_1"] = "OpenCV resizeWindow_1"
NODE_CLASS_MAPPINGS	["resizeWindow_1"] = cv2_resizeWindow_1

class cv2_rotate_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"rotateCode"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, rotateCode, dst=None):
		ret = apply_function(cv2.rotate, [src, rotateCode, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["rotate_0"] = "OpenCV rotate_0"
NODE_CLASS_MAPPINGS	["rotate_0"] = cv2_rotate_0

class cv2_rotate_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"rotateCode"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, rotateCode, dst=None):
		ret = apply_function(cv2.rotate, [src, rotateCode, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["rotate_1"] = "OpenCV rotate_1"
NODE_CLASS_MAPPINGS	["rotate_1"] = cv2_rotate_1

class cv2_rotatedRectangleIntersection_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"rect1"	: ("STRING",),
				"rect2"	: ("STRING",),
			},
			'optional': {
				"intersectingRegion"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, rect1, rect2, intersectingRegion=None):
		ret = apply_function(cv2.rotatedRectangleIntersection, [rect1, rect2, intersectingRegion], [2], [0, 1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["rotatedRectangleIntersection_0"] = "OpenCV rotatedRectangleIntersection_0"
NODE_CLASS_MAPPINGS	["rotatedRectangleIntersection_0"] = cv2_rotatedRectangleIntersection_0

class cv2_rotatedRectangleIntersection_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"rect1"	: ("STRING",),
				"rect2"	: ("STRING",),
			},
			'optional': {
				"intersectingRegion"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, rect1, rect2, intersectingRegion=None):
		ret = apply_function(cv2.rotatedRectangleIntersection, [rect1, rect2, intersectingRegion], [2], [0, 1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["rotatedRectangleIntersection_1"] = "OpenCV rotatedRectangleIntersection_1"
NODE_CLASS_MAPPINGS	["rotatedRectangleIntersection_1"] = cv2_rotatedRectangleIntersection_1

class cv2_sampsonDistance_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"pt1"	: ("NPARRAY",),
				"pt2"	: ("NPARRAY",),
				"F"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, pt1, pt2, F):
		ret = apply_function(cv2.sampsonDistance, [pt1, pt2, F], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sampsonDistance_0"] = "OpenCV sampsonDistance_0"
NODE_CLASS_MAPPINGS	["sampsonDistance_0"] = cv2_sampsonDistance_0

class cv2_sampsonDistance_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"pt1"	: ("NPARRAY",),
				"pt2"	: ("NPARRAY",),
				"F"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("FLOAT",)
	RETURN_NAMES	= ("float",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, pt1, pt2, F):
		ret = apply_function(cv2.sampsonDistance, [pt1, pt2, F], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sampsonDistance_1"] = "OpenCV sampsonDistance_1"
NODE_CLASS_MAPPINGS	["sampsonDistance_1"] = cv2_sampsonDistance_1

class cv2_scaleAdd_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, alpha, src2, dst=None):
		ret = apply_function(cv2.scaleAdd, [src1, alpha, src2, dst], [0, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["scaleAdd_0"] = "OpenCV scaleAdd_0"
NODE_CLASS_MAPPINGS	["scaleAdd_0"] = cv2_scaleAdd_0

class cv2_scaleAdd_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"alpha"	: ("FLOAT",),
				"src2"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, alpha, src2, dst=None):
		ret = apply_function(cv2.scaleAdd, [src1, alpha, src2, dst], [0, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["scaleAdd_1"] = "OpenCV scaleAdd_1"
NODE_CLASS_MAPPINGS	["scaleAdd_1"] = cv2_scaleAdd_1

class cv2_seamlessClone_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"p"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"blend"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, mask, p, flags, blend=None):
		ret = apply_function(cv2.seamlessClone, [src, dst, mask, p, flags, blend], [0, 1, 2, 5], [3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["seamlessClone_0"] = "OpenCV seamlessClone_0"
NODE_CLASS_MAPPINGS	["seamlessClone_0"] = cv2_seamlessClone_0

class cv2_seamlessClone_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"p"	: ("STRING",),
				"flags"	: ("INT",),
			},
			'optional': {
				"blend"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst, mask, p, flags, blend=None):
		ret = apply_function(cv2.seamlessClone, [src, dst, mask, p, flags, blend], [0, 1, 2, 5], [3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["seamlessClone_1"] = "OpenCV seamlessClone_1"
NODE_CLASS_MAPPINGS	["seamlessClone_1"] = cv2_seamlessClone_1

class cv2_selectROI_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"windowName"	: ("STRING",),
				"img"	: ("NPARRAY",),
				"showCrosshair"	: ("BOOLEAN",),
				"fromCenter"	: ("BOOLEAN",),
				"printNotice"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, windowName, img, showCrosshair, fromCenter, printNotice):
		ret = apply_function(cv2.selectROI, [windowName, img, showCrosshair, fromCenter, printNotice], [1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["selectROI_0"] = "OpenCV selectROI_0"
NODE_CLASS_MAPPINGS	["selectROI_0"] = cv2_selectROI_0

class cv2_selectROI_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"windowName"	: ("STRING",),
				"img"	: ("NPARRAY",),
				"showCrosshair"	: ("BOOLEAN",),
				"fromCenter"	: ("BOOLEAN",),
				"printNotice"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, windowName, img, showCrosshair, fromCenter, printNotice):
		ret = apply_function(cv2.selectROI, [windowName, img, showCrosshair, fromCenter, printNotice], [1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["selectROI_1"] = "OpenCV selectROI_1"
NODE_CLASS_MAPPINGS	["selectROI_1"] = cv2_selectROI_1

class cv2_selectROI_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"showCrosshair"	: ("BOOLEAN",),
				"fromCenter"	: ("BOOLEAN",),
				"printNotice"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, showCrosshair, fromCenter, printNotice):
		ret = apply_function(cv2.selectROI, [img, showCrosshair, fromCenter, printNotice], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["selectROI_2"] = "OpenCV selectROI_2"
NODE_CLASS_MAPPINGS	["selectROI_2"] = cv2_selectROI_2

class cv2_selectROI_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"img"	: ("NPARRAY",),
				"showCrosshair"	: ("BOOLEAN",),
				"fromCenter"	: ("BOOLEAN",),
				"printNotice"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, img, showCrosshair, fromCenter, printNotice):
		ret = apply_function(cv2.selectROI, [img, showCrosshair, fromCenter, printNotice], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["selectROI_3"] = "OpenCV selectROI_3"
NODE_CLASS_MAPPINGS	["selectROI_3"] = cv2_selectROI_3

class cv2_sepFilter2D_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"kernelX"	: ("NPARRAY",),
				"kernelY"	: ("NPARRAY",),
				"anchor"	: ("STRING",),
				"delta"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, kernelX, kernelY, anchor, delta, borderType, dst=None):
		ret = apply_function(cv2.sepFilter2D, [src, ddepth, kernelX, kernelY, dst, anchor, delta, borderType], [0, 2, 3, 4], [5])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sepFilter2D_0"] = "OpenCV sepFilter2D_0"
NODE_CLASS_MAPPINGS	["sepFilter2D_0"] = cv2_sepFilter2D_0

class cv2_sepFilter2D_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"kernelX"	: ("NPARRAY",),
				"kernelY"	: ("NPARRAY",),
				"anchor"	: ("STRING",),
				"delta"	: ("FLOAT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, kernelX, kernelY, anchor, delta, borderType, dst=None):
		ret = apply_function(cv2.sepFilter2D, [src, ddepth, kernelX, kernelY, dst, anchor, delta, borderType], [0, 2, 3, 4], [5])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sepFilter2D_1"] = "OpenCV sepFilter2D_1"
NODE_CLASS_MAPPINGS	["sepFilter2D_1"] = cv2_sepFilter2D_1

class cv2_setIdentity_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"mtx"	: ("NPARRAY",),
				"s"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, mtx, s):
		ret = apply_function(cv2.setIdentity, [mtx, s], [0], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setIdentity_0"] = "OpenCV setIdentity_0"
NODE_CLASS_MAPPINGS	["setIdentity_0"] = cv2_setIdentity_0

class cv2_setIdentity_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"mtx"	: ("NPARRAY",),
				"s"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, mtx, s):
		ret = apply_function(cv2.setIdentity, [mtx, s], [0], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setIdentity_1"] = "OpenCV setIdentity_1"
NODE_CLASS_MAPPINGS	["setIdentity_1"] = cv2_setIdentity_1

class cv2_setLogLevel_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"level"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, level):
		ret = apply_function(cv2.setLogLevel, [level], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setLogLevel_0"] = "OpenCV setLogLevel_0"
NODE_CLASS_MAPPINGS	["setLogLevel_0"] = cv2_setLogLevel_0

class cv2_setNumThreads_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"nthreads"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, nthreads):
		ret = apply_function(cv2.setNumThreads, [nthreads], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setNumThreads_0"] = "OpenCV setNumThreads_0"
NODE_CLASS_MAPPINGS	["setNumThreads_0"] = cv2_setNumThreads_0

class cv2_setRNGSeed_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"seed"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, seed):
		ret = apply_function(cv2.setRNGSeed, [seed], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setRNGSeed_0"] = "OpenCV setRNGSeed_0"
NODE_CLASS_MAPPINGS	["setRNGSeed_0"] = cv2_setRNGSeed_0

class cv2_setTrackbarMax_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"trackbarname"	: ("STRING",),
				"winname"	: ("STRING",),
				"maxval"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, trackbarname, winname, maxval):
		ret = apply_function(cv2.setTrackbarMax, [trackbarname, winname, maxval], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setTrackbarMax_0"] = "OpenCV setTrackbarMax_0"
NODE_CLASS_MAPPINGS	["setTrackbarMax_0"] = cv2_setTrackbarMax_0

class cv2_setTrackbarMin_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"trackbarname"	: ("STRING",),
				"winname"	: ("STRING",),
				"minval"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, trackbarname, winname, minval):
		ret = apply_function(cv2.setTrackbarMin, [trackbarname, winname, minval], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setTrackbarMin_0"] = "OpenCV setTrackbarMin_0"
NODE_CLASS_MAPPINGS	["setTrackbarMin_0"] = cv2_setTrackbarMin_0

class cv2_setTrackbarPos_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"trackbarname"	: ("STRING",),
				"winname"	: ("STRING",),
				"pos"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, trackbarname, winname, pos):
		ret = apply_function(cv2.setTrackbarPos, [trackbarname, winname, pos], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setTrackbarPos_0"] = "OpenCV setTrackbarPos_0"
NODE_CLASS_MAPPINGS	["setTrackbarPos_0"] = cv2_setTrackbarPos_0

class cv2_setUseOpenVX_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"flag"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, flag):
		ret = apply_function(cv2.setUseOpenVX, [flag], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setUseOpenVX_0"] = "OpenCV setUseOpenVX_0"
NODE_CLASS_MAPPINGS	["setUseOpenVX_0"] = cv2_setUseOpenVX_0

class cv2_setUseOptimized_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"onoff"	: ("BOOLEAN",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, onoff):
		ret = apply_function(cv2.setUseOptimized, [onoff], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setUseOptimized_0"] = "OpenCV setUseOptimized_0"
NODE_CLASS_MAPPINGS	["setUseOptimized_0"] = cv2_setUseOptimized_0

class cv2_setWindowProperty_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"prop_id"	: ("INT",),
				"prop_value"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, prop_id, prop_value):
		ret = apply_function(cv2.setWindowProperty, [winname, prop_id, prop_value], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setWindowProperty_0"] = "OpenCV setWindowProperty_0"
NODE_CLASS_MAPPINGS	["setWindowProperty_0"] = cv2_setWindowProperty_0

class cv2_setWindowTitle_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"winname"	: ("STRING",),
				"title"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, winname, title):
		ret = apply_function(cv2.setWindowTitle, [winname, title], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["setWindowTitle_0"] = "OpenCV setWindowTitle_0"
NODE_CLASS_MAPPINGS	["setWindowTitle_0"] = cv2_setWindowTitle_0

class cv2_solve_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, flags, dst=None):
		ret = apply_function(cv2.solve, [src1, src2, dst, flags], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solve_0"] = "OpenCV solve_0"
NODE_CLASS_MAPPINGS	["solve_0"] = cv2_solve_0

class cv2_solve_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, flags, dst=None):
		ret = apply_function(cv2.solve, [src1, src2, dst, flags], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solve_1"] = "OpenCV solve_1"
NODE_CLASS_MAPPINGS	["solve_1"] = cv2_solve_1

class cv2_solveCubic_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"coeffs"	: ("NPARRAY",),
			},
			'optional': {
				"roots"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, coeffs, roots=None):
		ret = apply_function(cv2.solveCubic, [coeffs, roots], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solveCubic_0"] = "OpenCV solveCubic_0"
NODE_CLASS_MAPPINGS	["solveCubic_0"] = cv2_solveCubic_0

class cv2_solveCubic_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"coeffs"	: ("NPARRAY",),
			},
			'optional': {
				"roots"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, coeffs, roots=None):
		ret = apply_function(cv2.solveCubic, [coeffs, roots], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solveCubic_1"] = "OpenCV solveCubic_1"
NODE_CLASS_MAPPINGS	["solveCubic_1"] = cv2_solveCubic_1

class cv2_solveLP_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"Func"	: ("NPARRAY",),
				"Constr"	: ("NPARRAY",),
				"constr_eps"	: ("FLOAT",),
			},
			'optional': {
				"z"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, Func, Constr, constr_eps, z=None):
		ret = apply_function(cv2.solveLP, [Func, Constr, constr_eps, z], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solveLP_0"] = "OpenCV solveLP_0"
NODE_CLASS_MAPPINGS	["solveLP_0"] = cv2_solveLP_0

class cv2_solveLP_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"Func"	: ("NPARRAY",),
				"Constr"	: ("NPARRAY",),
				"constr_eps"	: ("FLOAT",),
			},
			'optional': {
				"z"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, Func, Constr, constr_eps, z=None):
		ret = apply_function(cv2.solveLP, [Func, Constr, constr_eps, z], [0, 1, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solveLP_1"] = "OpenCV solveLP_1"
NODE_CLASS_MAPPINGS	["solveLP_1"] = cv2_solveLP_1

class cv2_solveLP_2:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"Func"	: ("NPARRAY",),
				"Constr"	: ("NPARRAY",),
			},
			'optional': {
				"z"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, Func, Constr, z=None):
		ret = apply_function(cv2.solveLP, [Func, Constr, z], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solveLP_2"] = "OpenCV solveLP_2"
NODE_CLASS_MAPPINGS	["solveLP_2"] = cv2_solveLP_2

class cv2_solveLP_3:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"Func"	: ("NPARRAY",),
				"Constr"	: ("NPARRAY",),
			},
			'optional': {
				"z"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("INT", "NPARRAY",)
	RETURN_NAMES	= ("int", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, Func, Constr, z=None):
		ret = apply_function(cv2.solveLP, [Func, Constr, z], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solveLP_3"] = "OpenCV solveLP_3"
NODE_CLASS_MAPPINGS	["solveLP_3"] = cv2_solveLP_3

class cv2_solvePnP_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"objectPoints"	: ("NPARRAY",),
				"imagePoints"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"useExtrinsicGuess"	: ("BOOLEAN",),
				"flags"	: ("INT",),
			},
			'optional': {
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, objectPoints, imagePoints, cameraMatrix, distCoeffs, useExtrinsicGuess, flags, rvec=None, tvec=None):
		ret = apply_function(cv2.solvePnP, [objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, flags], [0, 1, 2, 3, 4, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solvePnP_0"] = "OpenCV solvePnP_0"
NODE_CLASS_MAPPINGS	["solvePnP_0"] = cv2_solvePnP_0

class cv2_solvePnP_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"objectPoints"	: ("NPARRAY",),
				"imagePoints"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"useExtrinsicGuess"	: ("BOOLEAN",),
				"flags"	: ("INT",),
			},
			'optional': {
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, objectPoints, imagePoints, cameraMatrix, distCoeffs, useExtrinsicGuess, flags, rvec=None, tvec=None):
		ret = apply_function(cv2.solvePnP, [objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, flags], [0, 1, 2, 3, 4, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solvePnP_1"] = "OpenCV solvePnP_1"
NODE_CLASS_MAPPINGS	["solvePnP_1"] = cv2_solvePnP_1

class cv2_solvePnPRansac_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"objectPoints"	: ("NPARRAY",),
				"imagePoints"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"useExtrinsicGuess"	: ("BOOLEAN",),
				"iterationsCount"	: ("INT",),
				"reprojectionError"	: ("FLOAT",),
				"confidence"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
				"inliers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray_1", "nparray_2", "nparray_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, objectPoints, imagePoints, cameraMatrix, distCoeffs, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, flags, rvec=None, tvec=None, inliers=None):
		ret = apply_function(cv2.solvePnPRansac, [objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers, flags], [0, 1, 2, 3, 4, 5, 10], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solvePnPRansac_0"] = "OpenCV solvePnPRansac_0"
NODE_CLASS_MAPPINGS	["solvePnPRansac_0"] = cv2_solvePnPRansac_0

class cv2_solvePnPRansac_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"objectPoints"	: ("NPARRAY",),
				"imagePoints"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"useExtrinsicGuess"	: ("BOOLEAN",),
				"iterationsCount"	: ("INT",),
				"reprojectionError"	: ("FLOAT",),
				"confidence"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
				"inliers"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray_1", "nparray_2", "nparray_3",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, objectPoints, imagePoints, cameraMatrix, distCoeffs, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, flags, rvec=None, tvec=None, inliers=None):
		ret = apply_function(cv2.solvePnPRansac, [objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers, flags], [0, 1, 2, 3, 4, 5, 10], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solvePnPRansac_1"] = "OpenCV solvePnPRansac_1"
NODE_CLASS_MAPPINGS	["solvePnPRansac_1"] = cv2_solvePnPRansac_1

class cv2_solvePnPRefineLM_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"objectPoints"	: ("NPARRAY",),
				"imagePoints"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, criteria):
		ret = apply_function(cv2.solvePnPRefineLM, [objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, criteria], [0, 1, 2, 3, 4, 5], [6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solvePnPRefineLM_0"] = "OpenCV solvePnPRefineLM_0"
NODE_CLASS_MAPPINGS	["solvePnPRefineLM_0"] = cv2_solvePnPRefineLM_0

class cv2_solvePnPRefineLM_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"objectPoints"	: ("NPARRAY",),
				"imagePoints"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, criteria):
		ret = apply_function(cv2.solvePnPRefineLM, [objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, criteria], [0, 1, 2, 3, 4, 5], [6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solvePnPRefineLM_1"] = "OpenCV solvePnPRefineLM_1"
NODE_CLASS_MAPPINGS	["solvePnPRefineLM_1"] = cv2_solvePnPRefineLM_1

class cv2_solvePnPRefineVVS_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"objectPoints"	: ("NPARRAY",),
				"imagePoints"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
				"criteria"	: ("STRING",),
				"VVSlambda"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, criteria, VVSlambda):
		ret = apply_function(cv2.solvePnPRefineVVS, [objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, criteria, VVSlambda], [0, 1, 2, 3, 4, 5], [6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solvePnPRefineVVS_0"] = "OpenCV solvePnPRefineVVS_0"
NODE_CLASS_MAPPINGS	["solvePnPRefineVVS_0"] = cv2_solvePnPRefineVVS_0

class cv2_solvePnPRefineVVS_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"objectPoints"	: ("NPARRAY",),
				"imagePoints"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"rvec"	: ("NPARRAY",),
				"tvec"	: ("NPARRAY",),
				"criteria"	: ("STRING",),
				"VVSlambda"	: ("FLOAT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, criteria, VVSlambda):
		ret = apply_function(cv2.solvePnPRefineVVS, [objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, criteria, VVSlambda], [0, 1, 2, 3, 4, 5], [6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solvePnPRefineVVS_1"] = "OpenCV solvePnPRefineVVS_1"
NODE_CLASS_MAPPINGS	["solvePnPRefineVVS_1"] = cv2_solvePnPRefineVVS_1

class cv2_solvePoly_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"coeffs"	: ("NPARRAY",),
				"maxIters"	: ("INT",),
			},
			'optional': {
				"roots"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, coeffs, maxIters, roots=None):
		ret = apply_function(cv2.solvePoly, [coeffs, roots, maxIters], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solvePoly_0"] = "OpenCV solvePoly_0"
NODE_CLASS_MAPPINGS	["solvePoly_0"] = cv2_solvePoly_0

class cv2_solvePoly_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"coeffs"	: ("NPARRAY",),
				"maxIters"	: ("INT",),
			},
			'optional': {
				"roots"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, coeffs, maxIters, roots=None):
		ret = apply_function(cv2.solvePoly, [coeffs, roots, maxIters], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["solvePoly_1"] = "OpenCV solvePoly_1"
NODE_CLASS_MAPPINGS	["solvePoly_1"] = cv2_solvePoly_1

class cv2_sort_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, dst=None):
		ret = apply_function(cv2.sort, [src, flags, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sort_0"] = "OpenCV sort_0"
NODE_CLASS_MAPPINGS	["sort_0"] = cv2_sort_0

class cv2_sort_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, dst=None):
		ret = apply_function(cv2.sort, [src, flags, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sort_1"] = "OpenCV sort_1"
NODE_CLASS_MAPPINGS	["sort_1"] = cv2_sort_1

class cv2_sortIdx_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, dst=None):
		ret = apply_function(cv2.sortIdx, [src, flags, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sortIdx_0"] = "OpenCV sortIdx_0"
NODE_CLASS_MAPPINGS	["sortIdx_0"] = cv2_sortIdx_0

class cv2_sortIdx_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, flags, dst=None):
		ret = apply_function(cv2.sortIdx, [src, flags, dst], [0, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sortIdx_1"] = "OpenCV sortIdx_1"
NODE_CLASS_MAPPINGS	["sortIdx_1"] = cv2_sortIdx_1

class cv2_spatialGradient_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("INT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dx"	: ("NPARRAY",),
				"dy"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, borderType, dx=None, dy=None):
		ret = apply_function(cv2.spatialGradient, [src, dx, dy, ksize, borderType], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["spatialGradient_0"] = "OpenCV spatialGradient_0"
NODE_CLASS_MAPPINGS	["spatialGradient_0"] = cv2_spatialGradient_0

class cv2_spatialGradient_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("INT",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dx"	: ("NPARRAY",),
				"dy"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("nparray_0", "nparray_1",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, borderType, dx=None, dy=None):
		ret = apply_function(cv2.spatialGradient, [src, dx, dy, ksize, borderType], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["spatialGradient_1"] = "OpenCV spatialGradient_1"
NODE_CLASS_MAPPINGS	["spatialGradient_1"] = cv2_spatialGradient_1

class cv2_sqrBoxFilter_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"ksize"	: ("STRING",),
				"anchor"	: ("STRING",),
				"normalize"	: ("BOOLEAN",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, ksize, anchor, normalize, borderType, dst=None):
		ret = apply_function(cv2.sqrBoxFilter, [src, ddepth, ksize, dst, anchor, normalize, borderType], [0, 3], [2, 4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sqrBoxFilter_0"] = "OpenCV sqrBoxFilter_0"
NODE_CLASS_MAPPINGS	["sqrBoxFilter_0"] = cv2_sqrBoxFilter_0

class cv2_sqrBoxFilter_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ddepth"	: ("INT",),
				"ksize"	: ("STRING",),
				"anchor"	: ("STRING",),
				"normalize"	: ("BOOLEAN",),
				"borderType"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ddepth, ksize, anchor, normalize, borderType, dst=None):
		ret = apply_function(cv2.sqrBoxFilter, [src, ddepth, ksize, dst, anchor, normalize, borderType], [0, 3], [2, 4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sqrBoxFilter_1"] = "OpenCV sqrBoxFilter_1"
NODE_CLASS_MAPPINGS	["sqrBoxFilter_1"] = cv2_sqrBoxFilter_1

class cv2_sqrt_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.sqrt, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sqrt_0"] = "OpenCV sqrt_0"
NODE_CLASS_MAPPINGS	["sqrt_0"] = cv2_sqrt_0

class cv2_sqrt_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.sqrt, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sqrt_1"] = "OpenCV sqrt_1"
NODE_CLASS_MAPPINGS	["sqrt_1"] = cv2_sqrt_1

class cv2_stackBlur_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, dst=None):
		ret = apply_function(cv2.stackBlur, [src, ksize, dst], [0, 2], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["stackBlur_0"] = "OpenCV stackBlur_0"
NODE_CLASS_MAPPINGS	["stackBlur_0"] = cv2_stackBlur_0

class cv2_stackBlur_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"ksize"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, ksize, dst=None):
		ret = apply_function(cv2.stackBlur, [src, ksize, dst], [0, 2], [1])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["stackBlur_1"] = "OpenCV stackBlur_1"
NODE_CLASS_MAPPINGS	["stackBlur_1"] = cv2_stackBlur_1

class cv2_startWindowThread_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.startWindowThread, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["startWindowThread_0"] = "OpenCV startWindowThread_0"
NODE_CLASS_MAPPINGS	["startWindowThread_0"] = cv2_startWindowThread_0

class cv2_stereoRectify_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix1"	: ("NPARRAY",),
				"distCoeffs1"	: ("NPARRAY",),
				"cameraMatrix2"	: ("NPARRAY",),
				"distCoeffs2"	: ("NPARRAY",),
				"imageSize"	: ("STRING",),
				"R"	: ("NPARRAY",),
				"T"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"alpha"	: ("FLOAT",),
				"newImageSize"	: ("STRING",),
			},
			'optional': {
				"R1"	: ("NPARRAY",),
				"R2"	: ("NPARRAY",),
				"P1"	: ("NPARRAY",),
				"P2"	: ("NPARRAY",),
				"Q"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "STRING", "STRING",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2", "nparray_3", "nparray_4", "literal_5", "literal_6",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, flags, alpha, newImageSize, R1=None, R2=None, P1=None, P2=None, Q=None):
		ret = apply_function(cv2.stereoRectify, [cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q, flags, alpha, newImageSize], [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11], [4, 14])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["stereoRectify_0"] = "OpenCV stereoRectify_0"
NODE_CLASS_MAPPINGS	["stereoRectify_0"] = cv2_stereoRectify_0

class cv2_stereoRectify_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"cameraMatrix1"	: ("NPARRAY",),
				"distCoeffs1"	: ("NPARRAY",),
				"cameraMatrix2"	: ("NPARRAY",),
				"distCoeffs2"	: ("NPARRAY",),
				"imageSize"	: ("STRING",),
				"R"	: ("NPARRAY",),
				"T"	: ("NPARRAY",),
				"flags"	: ("INT",),
				"alpha"	: ("FLOAT",),
				"newImageSize"	: ("STRING",),
			},
			'optional': {
				"R1"	: ("NPARRAY",),
				"R2"	: ("NPARRAY",),
				"P1"	: ("NPARRAY",),
				"P2"	: ("NPARRAY",),
				"Q"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "STRING", "STRING",)
	RETURN_NAMES	= ("nparray_0", "nparray_1", "nparray_2", "nparray_3", "nparray_4", "literal_5", "literal_6",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, flags, alpha, newImageSize, R1=None, R2=None, P1=None, P2=None, Q=None):
		ret = apply_function(cv2.stereoRectify, [cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q, flags, alpha, newImageSize], [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11], [4, 14])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["stereoRectify_1"] = "OpenCV stereoRectify_1"
NODE_CLASS_MAPPINGS	["stereoRectify_1"] = cv2_stereoRectify_1

class cv2_stereoRectifyUncalibrated_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"F"	: ("NPARRAY",),
				"imgSize"	: ("STRING",),
				"threshold"	: ("FLOAT",),
			},
			'optional': {
				"H1"	: ("NPARRAY",),
				"H2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, F, imgSize, threshold, H1=None, H2=None):
		ret = apply_function(cv2.stereoRectifyUncalibrated, [points1, points2, F, imgSize, H1, H2, threshold], [0, 1, 2, 4, 5], [3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["stereoRectifyUncalibrated_0"] = "OpenCV stereoRectifyUncalibrated_0"
NODE_CLASS_MAPPINGS	["stereoRectifyUncalibrated_0"] = cv2_stereoRectifyUncalibrated_0

class cv2_stereoRectifyUncalibrated_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"points1"	: ("NPARRAY",),
				"points2"	: ("NPARRAY",),
				"F"	: ("NPARRAY",),
				"imgSize"	: ("STRING",),
				"threshold"	: ("FLOAT",),
			},
			'optional': {
				"H1"	: ("NPARRAY",),
				"H2"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("BOOLEAN", "NPARRAY", "NPARRAY",)
	RETURN_NAMES	= ("bool", "nparray_1", "nparray_2",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, points1, points2, F, imgSize, threshold, H1=None, H2=None):
		ret = apply_function(cv2.stereoRectifyUncalibrated, [points1, points2, F, imgSize, H1, H2, threshold], [0, 1, 2, 4, 5], [3])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["stereoRectifyUncalibrated_1"] = "OpenCV stereoRectifyUncalibrated_1"
NODE_CLASS_MAPPINGS	["stereoRectifyUncalibrated_1"] = cv2_stereoRectifyUncalibrated_1

class cv2_stylization_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sigma_s"	: ("FLOAT",),
				"sigma_r"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sigma_s, sigma_r, dst=None):
		ret = apply_function(cv2.stylization, [src, dst, sigma_s, sigma_r], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["stylization_0"] = "OpenCV stylization_0"
NODE_CLASS_MAPPINGS	["stylization_0"] = cv2_stylization_0

class cv2_stylization_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"sigma_s"	: ("FLOAT",),
				"sigma_r"	: ("FLOAT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, sigma_s, sigma_r, dst=None):
		ret = apply_function(cv2.stylization, [src, dst, sigma_s, sigma_r], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["stylization_1"] = "OpenCV stylization_1"
NODE_CLASS_MAPPINGS	["stylization_1"] = cv2_stylization_1

class cv2_subtract_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dtype, dst=None, mask=None):
		ret = apply_function(cv2.subtract, [src1, src2, dst, mask, dtype], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["subtract_0"] = "OpenCV subtract_0"
NODE_CLASS_MAPPINGS	["subtract_0"] = cv2_subtract_0

class cv2_subtract_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src1"	: ("NPARRAY",),
				"src2"	: ("NPARRAY",),
				"dtype"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src1, src2, dtype, dst=None, mask=None):
		ret = apply_function(cv2.subtract, [src1, src2, dst, mask, dtype], [0, 1, 2, 3], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["subtract_1"] = "OpenCV subtract_1"
NODE_CLASS_MAPPINGS	["subtract_1"] = cv2_subtract_1

class cv2_sumElems_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src):
		ret = apply_function(cv2.sumElems, [src], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sumElems_0"] = "OpenCV sumElems_0"
NODE_CLASS_MAPPINGS	["sumElems_0"] = cv2_sumElems_0

class cv2_sumElems_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src):
		ret = apply_function(cv2.sumElems, [src], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["sumElems_1"] = "OpenCV sumElems_1"
NODE_CLASS_MAPPINGS	["sumElems_1"] = cv2_sumElems_1

class cv2_textureFlattening_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"low_threshold"	: ("FLOAT",),
				"high_threshold"	: ("FLOAT",),
				"kernel_size"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask, low_threshold, high_threshold, kernel_size, dst=None):
		ret = apply_function(cv2.textureFlattening, [src, mask, dst, low_threshold, high_threshold, kernel_size], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["textureFlattening_0"] = "OpenCV textureFlattening_0"
NODE_CLASS_MAPPINGS	["textureFlattening_0"] = cv2_textureFlattening_0

class cv2_textureFlattening_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"mask"	: ("NPARRAY",),
				"low_threshold"	: ("FLOAT",),
				"high_threshold"	: ("FLOAT",),
				"kernel_size"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, mask, low_threshold, high_threshold, kernel_size, dst=None):
		ret = apply_function(cv2.textureFlattening, [src, mask, dst, low_threshold, high_threshold, kernel_size], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["textureFlattening_1"] = "OpenCV textureFlattening_1"
NODE_CLASS_MAPPINGS	["textureFlattening_1"] = cv2_textureFlattening_1

class cv2_threshold_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"thresh"	: ("FLOAT",),
				"maxval"	: ("FLOAT",),
				"type"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, thresh, maxval, type, dst=None):
		ret = apply_function(cv2.threshold, [src, thresh, maxval, type, dst], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["threshold_0"] = "OpenCV threshold_0"
NODE_CLASS_MAPPINGS	["threshold_0"] = cv2_threshold_0

class cv2_threshold_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"thresh"	: ("FLOAT",),
				"maxval"	: ("FLOAT",),
				"type"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("FLOAT", "NPARRAY",)
	RETURN_NAMES	= ("float", "nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, thresh, maxval, type, dst=None):
		ret = apply_function(cv2.threshold, [src, thresh, maxval, type, dst], [0, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["threshold_1"] = "OpenCV threshold_1"
NODE_CLASS_MAPPINGS	["threshold_1"] = cv2_threshold_1

class cv2_trace_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"mtx"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, mtx):
		ret = apply_function(cv2.trace, [mtx], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["trace_0"] = "OpenCV trace_0"
NODE_CLASS_MAPPINGS	["trace_0"] = cv2_trace_0

class cv2_trace_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"mtx"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("STRING",)
	RETURN_NAMES	= ("literal",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, mtx):
		ret = apply_function(cv2.trace, [mtx], [0], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["trace_1"] = "OpenCV trace_1"
NODE_CLASS_MAPPINGS	["trace_1"] = cv2_trace_1

class cv2_transform_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"m"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, m, dst=None):
		ret = apply_function(cv2.transform, [src, m, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["transform_0"] = "OpenCV transform_0"
NODE_CLASS_MAPPINGS	["transform_0"] = cv2_transform_0

class cv2_transform_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"m"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, m, dst=None):
		ret = apply_function(cv2.transform, [src, m, dst], [0, 1, 2], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["transform_1"] = "OpenCV transform_1"
NODE_CLASS_MAPPINGS	["transform_1"] = cv2_transform_1

class cv2_transpose_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.transpose, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["transpose_0"] = "OpenCV transpose_0"
NODE_CLASS_MAPPINGS	["transpose_0"] = cv2_transpose_0

class cv2_transpose_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dst=None):
		ret = apply_function(cv2.transpose, [src, dst], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["transpose_1"] = "OpenCV transpose_1"
NODE_CLASS_MAPPINGS	["transpose_1"] = cv2_transpose_1

class cv2_triangulatePoints_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"projMatr1"	: ("NPARRAY",),
				"projMatr2"	: ("NPARRAY",),
				"projPoints1"	: ("NPARRAY",),
				"projPoints2"	: ("NPARRAY",),
			},
			'optional': {
				"points4D"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, projMatr1, projMatr2, projPoints1, projPoints2, points4D=None):
		ret = apply_function(cv2.triangulatePoints, [projMatr1, projMatr2, projPoints1, projPoints2, points4D], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["triangulatePoints_0"] = "OpenCV triangulatePoints_0"
NODE_CLASS_MAPPINGS	["triangulatePoints_0"] = cv2_triangulatePoints_0

class cv2_triangulatePoints_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"projMatr1"	: ("NPARRAY",),
				"projMatr2"	: ("NPARRAY",),
				"projPoints1"	: ("NPARRAY",),
				"projPoints2"	: ("NPARRAY",),
			},
			'optional': {
				"points4D"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, projMatr1, projMatr2, projPoints1, projPoints2, points4D=None):
		ret = apply_function(cv2.triangulatePoints, [projMatr1, projMatr2, projPoints1, projPoints2, points4D], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["triangulatePoints_1"] = "OpenCV triangulatePoints_1"
NODE_CLASS_MAPPINGS	["triangulatePoints_1"] = cv2_triangulatePoints_1

class cv2_undistort_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"newCameraMatrix"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, cameraMatrix, distCoeffs, dst=None, newCameraMatrix=None):
		ret = apply_function(cv2.undistort, [src, cameraMatrix, distCoeffs, dst, newCameraMatrix], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["undistort_0"] = "OpenCV undistort_0"
NODE_CLASS_MAPPINGS	["undistort_0"] = cv2_undistort_0

class cv2_undistort_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"newCameraMatrix"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, cameraMatrix, distCoeffs, dst=None, newCameraMatrix=None):
		ret = apply_function(cv2.undistort, [src, cameraMatrix, distCoeffs, dst, newCameraMatrix], [0, 1, 2, 3, 4], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["undistort_1"] = "OpenCV undistort_1"
NODE_CLASS_MAPPINGS	["undistort_1"] = cv2_undistort_1

class cv2_undistortImagePoints_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"arg1"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, cameraMatrix, distCoeffs, arg1, dst=None):
		ret = apply_function(cv2.undistortImagePoints, [src, cameraMatrix, distCoeffs, dst, arg1], [0, 1, 2, 3], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["undistortImagePoints_0"] = "OpenCV undistortImagePoints_0"
NODE_CLASS_MAPPINGS	["undistortImagePoints_0"] = cv2_undistortImagePoints_0

class cv2_undistortImagePoints_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"arg1"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, cameraMatrix, distCoeffs, arg1, dst=None):
		ret = apply_function(cv2.undistortImagePoints, [src, cameraMatrix, distCoeffs, dst, arg1], [0, 1, 2, 3], [4])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["undistortImagePoints_1"] = "OpenCV undistortImagePoints_1"
NODE_CLASS_MAPPINGS	["undistortImagePoints_1"] = cv2_undistortImagePoints_1

class cv2_undistortPoints_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"R"	: ("NPARRAY",),
				"P"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, cameraMatrix, distCoeffs, dst=None, R=None, P=None):
		ret = apply_function(cv2.undistortPoints, [src, cameraMatrix, distCoeffs, dst, R, P], [0, 1, 2, 3, 4, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["undistortPoints_0"] = "OpenCV undistortPoints_0"
NODE_CLASS_MAPPINGS	["undistortPoints_0"] = cv2_undistortPoints_0

class cv2_undistortPoints_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
				"R"	: ("NPARRAY",),
				"P"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, cameraMatrix, distCoeffs, dst=None, R=None, P=None):
		ret = apply_function(cv2.undistortPoints, [src, cameraMatrix, distCoeffs, dst, R, P], [0, 1, 2, 3, 4, 5], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["undistortPoints_1"] = "OpenCV undistortPoints_1"
NODE_CLASS_MAPPINGS	["undistortPoints_1"] = cv2_undistortPoints_1

class cv2_undistortPointsIter_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"R"	: ("NPARRAY",),
				"P"	: ("NPARRAY",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, cameraMatrix, distCoeffs, R, P, criteria, dst=None):
		ret = apply_function(cv2.undistortPointsIter, [src, cameraMatrix, distCoeffs, R, P, criteria, dst], [0, 1, 2, 3, 4, 6], [5])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["undistortPointsIter_0"] = "OpenCV undistortPointsIter_0"
NODE_CLASS_MAPPINGS	["undistortPointsIter_0"] = cv2_undistortPointsIter_0

class cv2_undistortPointsIter_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"cameraMatrix"	: ("NPARRAY",),
				"distCoeffs"	: ("NPARRAY",),
				"R"	: ("NPARRAY",),
				"P"	: ("NPARRAY",),
				"criteria"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, cameraMatrix, distCoeffs, R, P, criteria, dst=None):
		ret = apply_function(cv2.undistortPointsIter, [src, cameraMatrix, distCoeffs, R, P, criteria, dst], [0, 1, 2, 3, 4, 6], [5])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["undistortPointsIter_1"] = "OpenCV undistortPointsIter_1"
NODE_CLASS_MAPPINGS	["undistortPointsIter_1"] = cv2_undistortPointsIter_1

class cv2_useOpenVX_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.useOpenVX, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["useOpenVX_0"] = "OpenCV useOpenVX_0"
NODE_CLASS_MAPPINGS	["useOpenVX_0"] = cv2_useOpenVX_0

class cv2_useOptimized_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, ):
		ret = apply_function(cv2.useOptimized, [], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["useOptimized_0"] = "OpenCV useOptimized_0"
NODE_CLASS_MAPPINGS	["useOptimized_0"] = cv2_useOptimized_0

class cv2_validateDisparity_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"disparity"	: ("NPARRAY",),
				"cost"	: ("NPARRAY",),
				"minDisparity"	: ("INT",),
				"numberOfDisparities"	: ("INT",),
				"disp12MaxDisp"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, disparity, cost, minDisparity, numberOfDisparities, disp12MaxDisp):
		ret = apply_function(cv2.validateDisparity, [disparity, cost, minDisparity, numberOfDisparities, disp12MaxDisp], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["validateDisparity_0"] = "OpenCV validateDisparity_0"
NODE_CLASS_MAPPINGS	["validateDisparity_0"] = cv2_validateDisparity_0

class cv2_validateDisparity_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"disparity"	: ("NPARRAY",),
				"cost"	: ("NPARRAY",),
				"minDisparity"	: ("INT",),
				"numberOfDisparities"	: ("INT",),
				"disp12MaxDisp"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, disparity, cost, minDisparity, numberOfDisparities, disp12MaxDisp):
		ret = apply_function(cv2.validateDisparity, [disparity, cost, minDisparity, numberOfDisparities, disp12MaxDisp], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["validateDisparity_1"] = "OpenCV validateDisparity_1"
NODE_CLASS_MAPPINGS	["validateDisparity_1"] = cv2_validateDisparity_1

class cv2_waitKey_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"delay"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, delay):
		ret = apply_function(cv2.waitKey, [delay], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["waitKey_0"] = "OpenCV waitKey_0"
NODE_CLASS_MAPPINGS	["waitKey_0"] = cv2_waitKey_0

class cv2_waitKeyEx_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"delay"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, delay):
		ret = apply_function(cv2.waitKeyEx, [delay], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["waitKeyEx_0"] = "OpenCV waitKeyEx_0"
NODE_CLASS_MAPPINGS	["waitKeyEx_0"] = cv2_waitKeyEx_0

class cv2_warpAffine_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"M"	: ("NPARRAY",),
				"dsize"	: ("STRING",),
				"flags"	: ("INT",),
				"borderMode"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, M, dsize, flags, borderMode, borderValue, dst=None):
		ret = apply_function(cv2.warpAffine, [src, M, dsize, dst, flags, borderMode, borderValue], [0, 1, 3], [2, 6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["warpAffine_0"] = "OpenCV warpAffine_0"
NODE_CLASS_MAPPINGS	["warpAffine_0"] = cv2_warpAffine_0

class cv2_warpAffine_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"M"	: ("NPARRAY",),
				"dsize"	: ("STRING",),
				"flags"	: ("INT",),
				"borderMode"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, M, dsize, flags, borderMode, borderValue, dst=None):
		ret = apply_function(cv2.warpAffine, [src, M, dsize, dst, flags, borderMode, borderValue], [0, 1, 3], [2, 6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["warpAffine_1"] = "OpenCV warpAffine_1"
NODE_CLASS_MAPPINGS	["warpAffine_1"] = cv2_warpAffine_1

class cv2_warpPerspective_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"M"	: ("NPARRAY",),
				"dsize"	: ("STRING",),
				"flags"	: ("INT",),
				"borderMode"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, M, dsize, flags, borderMode, borderValue, dst=None):
		ret = apply_function(cv2.warpPerspective, [src, M, dsize, dst, flags, borderMode, borderValue], [0, 1, 3], [2, 6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["warpPerspective_0"] = "OpenCV warpPerspective_0"
NODE_CLASS_MAPPINGS	["warpPerspective_0"] = cv2_warpPerspective_0

class cv2_warpPerspective_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"M"	: ("NPARRAY",),
				"dsize"	: ("STRING",),
				"flags"	: ("INT",),
				"borderMode"	: ("INT",),
				"borderValue"	: ("STRING",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, M, dsize, flags, borderMode, borderValue, dst=None):
		ret = apply_function(cv2.warpPerspective, [src, M, dsize, dst, flags, borderMode, borderValue], [0, 1, 3], [2, 6])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["warpPerspective_1"] = "OpenCV warpPerspective_1"
NODE_CLASS_MAPPINGS	["warpPerspective_1"] = cv2_warpPerspective_1

class cv2_warpPolar_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dsize"	: ("STRING",),
				"center"	: ("STRING",),
				"maxRadius"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dsize, center, maxRadius, flags, dst=None):
		ret = apply_function(cv2.warpPolar, [src, dsize, center, maxRadius, flags, dst], [0, 5], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["warpPolar_0"] = "OpenCV warpPolar_0"
NODE_CLASS_MAPPINGS	["warpPolar_0"] = cv2_warpPolar_0

class cv2_warpPolar_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"src"	: ("NPARRAY",),
				"dsize"	: ("STRING",),
				"center"	: ("STRING",),
				"maxRadius"	: ("FLOAT",),
				"flags"	: ("INT",),
			},
			'optional': {
				"dst"	: ("NPARRAY",),
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, src, dsize, center, maxRadius, flags, dst=None):
		ret = apply_function(cv2.warpPolar, [src, dsize, center, maxRadius, flags, dst], [0, 5], [1, 2])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["warpPolar_1"] = "OpenCV warpPolar_1"
NODE_CLASS_MAPPINGS	["warpPolar_1"] = cv2_warpPolar_1

class cv2_watershed_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"markers"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, markers):
		ret = apply_function(cv2.watershed, [image, markers], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["watershed_0"] = "OpenCV watershed_0"
NODE_CLASS_MAPPINGS	["watershed_0"] = cv2_watershed_0

class cv2_watershed_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"image"	: ("NPARRAY",),
				"markers"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("NPARRAY",)
	RETURN_NAMES	= ("nparray",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, image, markers):
		ret = apply_function(cv2.watershed, [image, markers], [0, 1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["watershed_1"] = "OpenCV watershed_1"
NODE_CLASS_MAPPINGS	["watershed_1"] = cv2_watershed_1

class cv2_writeOpticalFlow_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"path"	: ("STRING",),
				"flow"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, path, flow):
		ret = apply_function(cv2.writeOpticalFlow, [path, flow], [1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["writeOpticalFlow_0"] = "OpenCV writeOpticalFlow_0"
NODE_CLASS_MAPPINGS	["writeOpticalFlow_0"] = cv2_writeOpticalFlow_0

class cv2_writeOpticalFlow_1:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"path"	: ("STRING",),
				"flow"	: ("NPARRAY",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("BOOLEAN",)
	RETURN_NAMES	= ("bool",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, path, flow):
		ret = apply_function(cv2.writeOpticalFlow, [path, flow], [1], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["writeOpticalFlow_1"] = "OpenCV writeOpticalFlow_1"
NODE_CLASS_MAPPINGS	["writeOpticalFlow_1"] = cv2_writeOpticalFlow_1

class cv2_CV_8UC_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"channels"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, channels):
		ret = apply_function(cv2.CV_8UC, [channels], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["CV_8UC_0"] = "OpenCV CV_8UC_0"
NODE_CLASS_MAPPINGS	["CV_8UC_0"] = cv2_CV_8UC_0

class cv2_CV_8SC_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"channels"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, channels):
		ret = apply_function(cv2.CV_8SC, [channels], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["CV_8SC_0"] = "OpenCV CV_8SC_0"
NODE_CLASS_MAPPINGS	["CV_8SC_0"] = cv2_CV_8SC_0

class cv2_CV_16UC_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"channels"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, channels):
		ret = apply_function(cv2.CV_16UC, [channels], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["CV_16UC_0"] = "OpenCV CV_16UC_0"
NODE_CLASS_MAPPINGS	["CV_16UC_0"] = cv2_CV_16UC_0

class cv2_CV_16SC_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"channels"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, channels):
		ret = apply_function(cv2.CV_16SC, [channels], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["CV_16SC_0"] = "OpenCV CV_16SC_0"
NODE_CLASS_MAPPINGS	["CV_16SC_0"] = cv2_CV_16SC_0

class cv2_CV_32SC_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"channels"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, channels):
		ret = apply_function(cv2.CV_32SC, [channels], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["CV_32SC_0"] = "OpenCV CV_32SC_0"
NODE_CLASS_MAPPINGS	["CV_32SC_0"] = cv2_CV_32SC_0

class cv2_CV_32FC_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"channels"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, channels):
		ret = apply_function(cv2.CV_32FC, [channels], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["CV_32FC_0"] = "OpenCV CV_32FC_0"
NODE_CLASS_MAPPINGS	["CV_32FC_0"] = cv2_CV_32FC_0

class cv2_CV_64FC_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"channels"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, channels):
		ret = apply_function(cv2.CV_64FC, [channels], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["CV_64FC_0"] = "OpenCV CV_64FC_0"
NODE_CLASS_MAPPINGS	["CV_64FC_0"] = cv2_CV_64FC_0

class cv2_CV_16FC_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"channels"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, channels):
		ret = apply_function(cv2.CV_16FC, [channels], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["CV_16FC_0"] = "OpenCV CV_16FC_0"
NODE_CLASS_MAPPINGS	["CV_16FC_0"] = cv2_CV_16FC_0

class cv2_CV_MAKETYPE_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"depth"	: ("INT",),
				"channels"	: ("INT",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("INT",)
	RETURN_NAMES	= ("int",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, depth, channels):
		ret = apply_function(cv2.CV_MAKETYPE, [depth, channels], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["CV_MAKETYPE_0"] = "OpenCV CV_MAKETYPE_0"
NODE_CLASS_MAPPINGS	["CV_MAKETYPE_0"] = cv2_CV_MAKETYPE_0

class cv2_dnn_unregisterLayer_0:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			'required': {
				"layerTypeName"	: ("STRING",),
			},
			'optional': {
				
			},
		}
	RETURN_TYPES	= ("None",)
	RETURN_NAMES	= ("unknown",)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, layerTypeName):
		ret = apply_function(cv2.dnn_unregisterLayer, [layerTypeName], [], [])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["dnn_unregisterLayer_0"] = "OpenCV dnn_unregisterLayer_0"
NODE_CLASS_MAPPINGS	["dnn_unregisterLayer_0"] = cv2_dnn_unregisterLayer_0
