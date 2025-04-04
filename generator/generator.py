import ast
import csv
import cv2
import os
from typing import List, TypedDict, Tuple
from itertools import zip_longest

class_prefix	= "cv2_"
display_prefix	= "OpenCV "

class FuncInfo(TypedDict):
	name: str
	params: List[Tuple[str, str]]
	return_type: str

image_types = [
	"UMat",
	"cv2.typing.MatLike",
	"cv2.typing.MatLike | None", # usually used for dst, but some dst also use non-none types. and as src type in reprojectImageTo3D
	"cv2.cuda.GpuMat" # only used in imshow
	#numpy.ndarray #only used in return type of imencode*
]

int_types = [
	"int",
	"DrawMatchesFlags",
	"HandEyeCalibrationMethod",
	"RobotWorldHandEyeCalibrationMethod",
	"SolvePnPMethod",
	"AlgorithmHint",
]

literal_types = [
	"cv2.typing.Moments",
	"cv2.typing.Point",
	"cv2.typing.Point2d",
	"cv2.typing.Point2f",
	"cv2.typing.Range",
	"cv2.typing.Rect",
	"cv2.typing.Rect2d",
	"cv2.typing.RotatedRect",
	"cv2.typing.Scalar",
	"cv2.typing.Size",
	"cv2.typing.TermCriteria",
]

unsupported_types = [
	"_typing.Sequence", # we could probably support this
	"_typing.Callable",
	"_typing.Type[cv2.dnn.LayerProtocol]", # class
	# classes
	"HistogramCostExtractor",
	"ShapeTransformer",
	"UsacParams",
	"Animation",
	"CirclesGridFinderParameters",
	"cv2.typing.FeatureDetector",
	"AffineTransformer"
	"AlignMTB"
	"BackgroundSubtractorKNN"
	"BackgroundSubtractorMOG2"
	"CalibrateDebevec"
	"CalibrateRobertson"
	"CLAHE"
	"GArrayDesc"
	"GeneralizedHoughBallard"
	"GeneralizedHoughGuil"
	"GOpaqueDesc"
	"GScalarDesc"
	"HausdorffDistanceExtractor"
	"HistogramCostExtractor"
	"LineSegmentDetector"
	"MergeDebevec"
	"MergeMertens"
	"MergeRobertson"
	"ShapeContextDistanceExtractor"
	"ThinPlateSplineShapeTransformer"
	"Tonemap"
	"TonemapDrago"
	"TonemapMantiuk"
	"TonemapReinhard"
]

node_def = '''
class {class_prefix}{class_name}:
	@classmethod
	def INPUT_TYPES(cls):
		return {{
			'required': {{
				{requireds}
			}},
			'optional': {{
				{optionals}
			}},
		}}
	RETURN_TYPES	= ({return_types},)
	RETURN_NAMES	= ({return_names},)
	FUNCTION	= 'execute'
	CATEGORY	= 'image/OpenCV'

	def execute(self, {header_args}):
		ret = apply_function(cv2.{func_name}, [{func_args}], [{image_idxs}], [{literal_idxs}])
		return ret

NODE_DISPLAY_NAME_MAPPINGS	["{class_name}"] = "{display_prefix}{class_name}"
NODE_CLASS_MAPPINGS	["{class_name}"] = {class_prefix}{class_name}
'''

def parse_functions(content) -> List[FuncInfo]:
	tree = ast.parse(content)

	functions = []

	for node in ast.iter_child_nodes(tree):
		if isinstance(node, ast.FunctionDef):
			# Skip methods (functions inside classes)
			if any(isinstance(parent, ast.ClassDef) for parent in ast.iter_child_nodes(tree)
				  if hasattr(parent, 'body') and node in parent.body):
				continue

			params = []
			for arg in node.args.args:
				param_name = arg.arg

				# Get parameter type
				param_type = "Any"
				if arg.annotation:
					if isinstance(arg.annotation, ast.Name):
						param_type = arg.annotation.id
					else:
						param_type = ast.unparse(arg.annotation)

				params.append((param_name, param_type))

			return_type = "Any"
			if node.returns:
				if isinstance(node.returns, ast.Name):
					return_type = node.returns.id
				elif isinstance(node.returns, ast.Subscript):
					return_type = ast.unparse(node.returns)
				else:
					return_type = ast.unparse(node.returns)

			func_info = FuncInfo(name=node.name, params=params, return_type=return_type)
			functions.append(func_info)

	return functions

def generate_nodes(func_infos: List[FuncInfo]):
	with open("generator/cv2base.py", 'r') as f:
		cv2base = "\n".join(f.read().splitlines()[1:-1])

	with open("cv2nodes.py", 'w') as f:
		f.write(cv2base)

		overload_counts	= {}
		overload_nos	= {}
		for func_info in func_infos:
			func_name = func_info["name"]
			if func_name in overload_counts:
				overload_counts[func_name] += 1
			else:
				overload_counts[func_name] = 1

		for func_info in func_infos:
			func_name = func_info["name"]
			overload_no = -1
			if overload_counts[func_name] > 0:
				if func_name in overload_nos:
					overload_no = overload_nos[func_name]
					overload_nos[func_name] += 1
				else:
					overload_no = 0
					overload_nos[func_name] = 1

			node_info = get_node_infos(func_info, overload_no)
			class_str = node_def.format(**node_info)
			f.write(class_str)

def type_starts_with(param_type: str, compare_types: List[str]) -> bool:
	for compare_type in compare_types:
		if param_type.startswith(compare_type): return True
	return False

def get_node_infos(func_info: FuncInfo, overload_no: int):
	func_name	= func_info["name"]
	params	= func_info["params"]
	return_type	= func_info["return_type"]

	comfy_requireds	= []
	comfy_optionals	= []
	func_requireds	= []
	func_optionals	= []
	func_args	= []
	image_idxs	= []
	literal_idxs	= []
	for idx, (param_name, param_type) in enumerate(params):
		if	type_starts_with(param_type, image_types)	: comfy_entry = f'"{param_name}"\t: ("NPARRAY",),'; image_idxs.append(idx)
		elif	type_starts_with(param_type, int_types)	: comfy_entry = f'"{param_name}"\t: ("INT",),'
		elif	param_type == "float"	: comfy_entry = f'"{param_name}"\t: ("FLOAT",),'
		elif	param_type == "bool"	: comfy_entry = f'"{param_name}"\t: ("BOOLEAN",),'
		elif	param_type == "str"	: comfy_entry = f'"{param_name}"\t: ("STRING",),'
		elif	param_type == "_typing.Any"	: comfy_entry = f'"{param_name}"\t: ("any",),' # see cv2base.py
		elif	type_starts_with(param_type, literal_types)	: comfy_entry = f'"{param_name}"\t: ("STRING",),'; literal_idxs.append(idx)

		if param_type.endswith("| None"):
			comfy_optionals	.append(comfy_entry)
			func_args	.append(param_name)
			func_optionals	.append(param_name)

		else:
			comfy_requireds	.append(comfy_entry)
			func_args	.append(param_name)
			func_requireds	.append(param_name)

	return_types = []
	return_names = []

	count_names = {}
	iter_types	= return_type[len("tuple["):-1].split(", ") if return_type.startswith("tuple") else [return_type]
	for iter_type in iter_types:
		return_type = '"None"'
		return_name = '"unknown"'

		if	type_starts_with(iter_type, image_types)	: return_type = '"NPARRAY"'	; return_name = '"nparray"'
		elif	type_starts_with(iter_type, int_types)	: return_type = '"INT"'	; return_name = '"int"'
		elif	iter_type == "float"	: return_type = '"FLOAT"'	; return_name = '"float"'
		elif	iter_type == "bool"	: return_type = '"BOOLEAN"'	; return_name = '"bool"'
		elif	iter_type == "str"	: return_type = '"STRING"'	; return_name = '"string"'
		elif	iter_type == "_typing.Any"	: return_type = '"any"'	; return_name = '"any"'
		elif	type_starts_with(iter_type, literal_types)	: return_type = '"STRING"'	; return_name = '"literal"'

		return_types.append(return_type)
		return_names.append(return_name)

		if return_name in count_names:
			count_names[return_name] += 1
		else:
			count_names[return_name] = 1

	for i, name in enumerate(return_names):
		if count_names[name] > 1:
			return_names[i] = f'{name[0:-1]}_{i}"'

	ret = {
		"class_name"	: func_name if overload_no == -1 else f'{func_name}_{overload_no}',
		"requireds"	: "\n\t\t\t\t".join(comfy_requireds),
		"optionals"	: "\n\t\t\t\t".join(comfy_optionals),
		"func_name"	: func_name,
		"header_args"	: ", ".join(func_requireds + [f'{param_name}=None' for param_name in func_optionals]),
		"func_args"	: ", ".join(func_args),
		"image_idxs"	: ", ".join(map(str, image_idxs)),
		"literal_idxs"	: ", ".join(map(str, literal_idxs)),
		"return_types"	: ", ".join(return_types),
		"return_names"	: ", ".join(return_names),
		"class_prefix"	: class_prefix,
		"display_prefix"	: display_prefix,
	}
	return ret

def analyze_functions(func_infos: List[FuncInfo], func_infos_supported: List[FuncInfo]):
	def get_signatures(func_infos: List[FuncInfo]) -> List[str]:
		return [f'{func_info["name"]}({", ".join([f"{key}: {value}" for key, value in func_info["params"]])}) -> {func_info["return_type"]}' for func_info in func_infos]

	print(f'{len(func_infos)} functions found, {len(func_infos_supported)} supported')

	# retrieved emperically after analyze
	images_types = [
		"_typing.Sequence[UMat]",
		"_typing.Sequence[cv2.typing.MatLiike]",
		"_typing.Sequence[cv2.typing.Rect]",
		"_typing.Sequence[cv2.typing.MatLike]",
		"_typing.Sequence[UMat] | None",
		]

	def get_unique_types(func_infos: List[FuncInfo]) -> Tuple[set, set]:
		params_unique = set()
		return_types_unique = set()

		for func_info in func_infos:
			# Collect unique parameter types
			for _, param_type in func_info['params']:
				params_unique.add(param_type)

			# Collect unique return types
			return_types_unique.add(func_info['return_type'])

		return list(params_unique), list(return_types_unique)

	params_types_unique, return_types_unique = get_unique_types(func_infos)
	print(f'{len(params_types_unique)} unique params types')
	#print("\n".join(params_types_unique) + f'\n\n{len(params_types_unique)} unique return types')

	print(f'{len(return_types_unique)} unique return types')
	#print("\n".join(return_types_unique) + f'\n\n{len(return_types_unique)} unique return types')

	# Function to check if a parameter is of an image type
	def is_image_type(param_type):
		return param_type in image_types

	# List to store functions with image types
	functions_with_image_params = []

	# Iterate over all functions and filter those with image types
	for func_info in func_infos:
		for name, type in func_info["params"]:
			if is_image_type(type):
				functions_with_image_params.append(func_info)
				break

	# List to store unique parameter names for the first image type
	first_image_param_names = set()

	# List to store unique parameter names for subsequent image types
	subsequent_image_param_names = set()

	# Iterate over functions with image parameters
	for func_info in functions_with_image_params:
		image_param_found = False
		for name, type in func_info["params"]:
			if is_image_type(type):
				if not image_param_found:
					first_image_param_names.add(name)
					image_param_found = True
				else:
					subsequent_image_param_names.add(name)

	# Print the results
	#print(f'\nUnique parameter names for the first image type: \n{" ".join(first_image_param_names)}')
	#print(f'\nUnique parameter names for subsequent image types: \n{" ".join(subsequent_image_param_names)}')

	# def randShuffle(dst: UMat, iterFactor: float = ...) -> UMat: ... # this will probably causes troubles
	transposed_data = zip_longest(
		["params_types_unique"]	+ params_types_unique,
		["return_types_unique"]	+ return_types_unique,
		["image_types"]	+ image_types,
		["images_types"]	+ images_types,
		["unsupported_types"]	+ unsupported_types,
		["first_image_param_names"]	+ list(first_image_param_names),
		["subsequent_image_param_names"]	+ list(subsequent_image_param_names),
		["signatures"]	+ get_signatures(func_infos),
		["signatures_filtered"]	+ get_signatures(func_infos_supported),
		fillvalue=''
	)

	with open('docs/cv2.tsv', mode='w', newline='') as file:
		writer = csv.writer(file, dialect="excel-tab")
		writer.writerows(transposed_data)

if __name__ == "__main__":
	cv2_path	= os.path.dirname	(cv2.__file__)
	pyi_path	= os.path.join	(cv2_path, "__init__.pyi") # standalone functions start at line 4825

	if not os.path.exists(pyi_path):
		raise FileNotFoundError(f'OpenCV .pyi file not found at "{pyi_path}"')

	with open(pyi_path, 'r') as f:
		content = f.read()

	func_infos = parse_functions(content) # 776

	func_infos_supported = [] # 656
	for func_info in func_infos:
		is_unsupported = False
		for unsupported_type in unsupported_types:
			if any(unsupported_type in func_type for _, func_type in func_info["params"]) or unsupported_type in func_info["return_type"]:
				is_unsupported = True
				break
		if is_unsupported: continue
		func_infos_supported.append(func_info)

	analyze_functions(func_infos, func_infos_supported)
	generate_nodes(func_infos_supported)
