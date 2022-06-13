import cv2
import numpy as np

def get_mask(img, mask):
	t = np.zeros(mask.shape)
	alpha = 1	
	beta = 0.2 # бета - прозрачность второй картинки
	gamma = 0

	def del_border(img, img_copy):
		test = np.zeros(img_copy.shape).astype(np.uint8)
		cnt, h = cv2.findContours(img_copy, 2, 1)
		for k in range(len(cnt)):
			cv2.drawContours(test, cnt, k, 255, -1)
		return test

	mask_copy = np.copy(mask).astype(np.uint8)
	contour_img = np.zeros(mask.shape).astype(np.uint8)

	hull_ar = None
	for i in range(0, 256, 8):
		mask_sl = mask_copy[:,i:i+8]
		contours, hierarchy = cv2.findContours(mask_sl, 2, 1)
		cnt = contours[0]
		for j in range(1, len(contours)):
			cnt = np.append(cnt, contours[j], axis=0)

		hull = cv2.convexHull(cnt)
		length = len(hull)
		for j in range(len(hull)):
			point1 = hull[j][0]
			point2 = hull[(j+1)%length][0]
			cv2.line(contour_img, (point1[0] + i, point1[1]), (point2[0] + i, point2[1]), 255, 1)

	mask_copy = del_border(mask, contour_img)
	beta = 0.5
	img_with_mask = cv2.addWeighted(img, alpha, cv2.cvtColor(mask_copy, cv2.COLOR_GRAY2RGB), beta, gamma)
	return {"img_with_mask": img_with_mask, "stack_mask": mask_copy}

def count_value(mask, stack_mask, def_size):
	width, height = mask.shape
	count_log, count_stack = 0, 0
	for i in range(width):
		for j in range(height):
			count_log += int(mask[i, j] == 255)
			count_stack += int(stack_mask[i, j] == 255)
	coeff = count_log / count_stack #Коэффициент полнодревесности
	cnt, h = cv2.findContours(mask, 2, 1)
	log_count = len(cnt) #Количество бревен
	mul_world = 25 #Длина эталона в пикселях
	mul_size = def_size[0] * def_size[1] / 256**2
	average_len = 6
	value = average_len * count_log * mul_size * (1 / (mul_world**2))
	print(count_log)
	print(def_size)
	return {"coeff": coeff, "log_count": log_count, "value": value }


