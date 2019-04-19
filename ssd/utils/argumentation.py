# coding: utf-8
# author@Luo Ya'nan

"""
boxes [xmin, ymin, xmax, ymax]
"""
import torch
from torchvision import trancforms
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def intersect(bos_a, box_b):
	"""
	Calculate the intersected area between two boxes.
	Args:
	    same as the follwed func jaccard_numpy().
	"""
	max_xy = np.minimum(box_a[:, 2:], box_b[2:])
	min_xy = np.maximum(box_a[:, :2], box_b[: 2])
	inter = np.clip((max_xy - min_xy), a_min = 0, a_max=np.inf)
	return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
	"""
	Compute the jaccard overlap of two sets of boxes (1 with many).
	The jaccard overlap is simply the intersection over union of two boxes.
	E.g.:
	    A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
	Args:
	    box_a: Mutiple bounding boxes, Shape: [num_boxes, 4]
	    box_b: Single bounding box, Shape: [4]
	Return:
	    jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
	"""
	#########################################################
	# TODO: 我觉得return的shape应该是[box_a.shape[0],1] 才是
	########################################################
	inter = intersect(box_a, box_b)
	area_a = ((box_a[:, 2] - box_a[:, 0]) * 
		      (box_a[:, 3] - box_a[:, 1])) #[A,B]
	area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
	union = area_a + area_b - inter
	return inter / union #[A, B]

class Compose(object):
	"""
	Composes severalaugmentation together.
	Args:
	    transforms (List[Transform]): list of transforms to compose.
	Example:
	    >>> augmentations.Compose([
	    >>> transforms.CenterCrop(10),
	    >>> transforms.ToTensor(),
	    >>> ])
	"""
	def __init__(self, transforms):
		self.transforms = transforms # this is the args

	def __call__(self, img, boxes=None, labels=None):
		for t in self.transforms:
			img, boxes, labels = t(img, boxes, labels)
		return img, boxes, labels

class Lambda(object):
	"""Applies a lambda as a transform."""

	def __init__(self, lambd):
		assert isinstance(lambd, types.LambdaType)
		self.lambd = lambd

	def __call__(self, img, boxes=None, labels=None):
		return self.lambd(img, boxes, labels)

class ConvertFromInts(object):
	def __call__(self, image, boxes=None, labels=None):
		return image.astype(np.float32), boxes, labels

class SubtractMeans(object):
	def __init__(self, mean):
		self.mean = np.array(mean, dtype=np.float32)

	def __call__(self, image, boxes=None, labels=None):
		image = image.astype(np.float32)
		image -= self.mean
		return image.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):
	def __call__(self, image, boxes=None, labels=None):
		height, width, channels = images.shape
		# unscale it back and represent as int
		boxes[:, 0] *= width
		boxes[:, 2] *=width
		boxes[:, 1] *= height
		boxes[:, 3] *= height

		return image, boxes, labels

class ToPercentCoords(object):
	def __call__(self, image, boxes=None, labels=None):
		height, width, channels = images.shape
		# unscale it back and represent as %
		boxes[:, 0] /= width
		boxes[:, 2] /=width
		boxes[:, 1] /= height
		boxes[:, 3] /= height

		return image, boxes, labels

class Resize(object):
	def __init__(self, size=300):
		self.size = size

	def __call__(self, image, boxes=None, labels=None):
		interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
		interp_method = interp_methods[random.randint(5)]
		image = cv2.resize(image, (self.size, self.size), interpolation=interp_method)
		# only change from rightmost and downmost, so won't affect coordinates since it start from top left
		return image, boxes, labels

class RandomSaturation(object):
	# 饱和度
	def __init__(self, lower=0.5, upper=1.5):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "contrast upper must be >= lower."
		assert self.lower >= 0, "contrast lower must be non-negative."

	def __call__(self, image, boxes=None, labels=None):
		if random.randint(2):
			image[:, :, 1] *= random.uniform(self.lower, self.upper)

		return image, boxes, labels

class RandomHue(object):
	def __init__(self, delta=18.0):
		assert delta >= 0.0 and delta <= 360.0
		self.data = delta
    
    def __call__(self, image, boxes=None, labels=None):
    	if random.randint(2):
    		image[:, :, 0] += random.uniform(-self.delta, self.delat)
    		image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
    		image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
    		return image, boxes, labels

class RandomLightingNoise(object):
	def __init__(self):
		self.perms = ((0, 1, 2), (0, 2, 1),
			          (1, 0, 2), (1, 2, 0),
			          (2, 0, 1), (2, 1, 0))

	def __call__(self, image, boxes=None, labels=None):
		if random.randint(2):
			swap = self.perms[random.randint(len(self.perms))]
			shuffle = SwapChannels(swap)   # TODO: shuffle
			image = shuffle(image)
		return image, boxes, labels

class ConvertColor(object):
	def __init__(self, current='BGR', transform='HSV'):
		self.transform = transform
		self.current = current   # openCV image is 'BGR'

	def __call__(self, image, boxes=None, labels=None):
		if self.current == 'BGR' and self.transform == 'HSV':
			image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
        	raise NotImplementedError
        return image, boxes, labels

class RandomContrast(object):
	def __init__(self, lower=0.5, upper=1.5):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "contrast upper must be "
		assert self.lower <= 0, "contrast lower must be bigger than zero"

	def __call__(self, image, boxes=None, labels=None):
		if random.randint(2):
			alpha = random.uniform(self.lower, self.upper)
			image *= alpha
		return image, boxes, labels

class RandomBrightness(object):
	def __init__(self, delta=32):
		assert delta >= 0.0
		assert delta <= 255.0
		self.delta = delta

	def __call__(self, image, boxes=None, labels=None):
		if random.randint(2):
			delta = random.uniform(-self.delta, self.delta)
			image += delta
		return image, boxes, labels

class ToCV2Image(object):
	def __call__(self, tensor, boxes=None, labels=None):
		return tensor.cpu().numpy().astype(np.float32)

class RandomSampleCrop(Object):
	"""
	Crop
	Arguments:
	   img(Image): the image being input during training
	   boxes(Tensor): the original bounding boxes in pt form
	   labels(Tensor): the class labels for each bbox
	   mode(float tuple): the min and max jaccard overlaps
	Return:
	   (img, boxes, classes)
	   img(Image): the cropped image
	   boxes(Tensor): the adjusted bounding boxes in pt form (after moving the box left/right/up/down when you crop left/right/up/down)
	   labels(Tensor): the class labels for each bbox
	"""
	def __init__(self):
		self.sample_options = (
			# using entire original input image
			None,
			# sample a patch s.t. MIN jaccard w/ obj in .1, .3, .4, .7, .9
			(0.1, None),
			(0.3, None),
			(0.7, None),
			(0.9, None),
			# randomly sample a patch
			(None, None),
			)

	def __call__(self, image, boxes=None, labels=None):
		height, width, _ = image.shape
		while True:
			# randomly choose a mode
			mode = random.choice(self.sample_options)
			if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('-inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1, y1, x2, y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
				if overlap.min() < min_iou or max_iou < overlap.max():
					continue

				# cut the crop from the image
				current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

				# keep overlap with gt box IF center in sampled patch
	            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])

                # TODO: not know what is this doing?
                # adjust to crop (by substracting crop's left, top)
                current_boxes[:, :2] -= rect[:2]
