import numpy as np
from scipy import stats

import cv2
from sklearn.cluster import DBSCAN

# model - обученная модель для сегментации (см. Приложение 2)
# class_areas - список относительных площадей классов

# цвета
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
color = [blue, green, red]


def get_relative_area(mask):
	return mask.sum() / mask.shape[0] / mask.shape[1]


def get_predict(original_img, clipping=0.5):
	img = cv2.resize(original_img / 255., (img_shape, img_shape))
	pred = model.predict(np.array([img]))[0]
	return pred


def get_dbscan_mask(mask, eps=1):
	x, y = np.where(mask == 1)

	arr = np.empty((x.shape[0], 2), dtype=np.uint(8))
	arr[:, 0] = x
	arr[:, 1] = y

	clustering = DBSCAN(eps=eps, min_samples=1).fit(arr)

	new_mask = np.zeros(mask.shape, dtype=np.uint8)
	for i in range(len(arr)):
		xx, yy = arr[i]
		new_mask[xx, yy] = clustering.labels_[i] + 1
	return new_mask


def get_dbscan_filt_mask(db_mask, class_index, min_area_coef=1 / 10.):
	min_area = class_area[class_index] * min_area_coef

	label, count = np.unique(db_mask, return_counts=True)

	delete = []
	obj_labels = []
	for i in range(1, len(label)):
		if count[i] / db_mask.shape[0] * db_mask.shape[1] < min_area:
			delete.append(label[i])
		else:
			obj_labels.append(label[i])

	new_mask = db_mask
	for class_index in delete:
		new_mask = np.where(new_mask != class_index, new_mask, 0)
	return new_mask, obj_labels


def get_border(mask, index=1):
	x, y = np.where(mask == index)

	down = x.min()
	up = x.max()

	left = y.min()
	right = y.max()

	return down, up, left, right


def get_detection_img(original_img, clipping=0.5, class_area_coef=0.5,
                      db_eps=1, min_area_coef=1 / 10.):
	h, w, c = original_img.shape

	pred = get_predict(original_img, clipping)

	new_img = original_img.copy()

	index_to_color_borders_dict = dict()
	free_color = [0, 1, 2]
	for i in range(1, classes_num):

		mask = pred[:, :, i]

		if get_relative_area(mask) < class_area[i] * class_area_coef:
			continue

		mask = (mask > clipping).astype(np.uint8)
		mask = get_dbscan_mask(mask, db_eps)
		mask, obj_labels = get_dbscan_filt_mask(mask, class_index=i, min_area_coef=min_area_coef)

		mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

		index_to_color_borders_dict[i] = [0, []]

		for obj in obj_labels:
			index_to_color_borders_dict[i][1].append(get_border(mask, obj))

		mask = np.where(mask == 0, mask, 1)
		neg = 1 - mask

		color_index = i % 3
		if color_index in free_color:
			free_color.remove(color_index)
			if len(free_color) == 0:
				free_color = [0, 1, 2]
		else:
			color_index = free_color[0]
		index_to_color_borders_dict[i][0] = color_index

		free = [0, 1, 2]
		free.remove(color_index)
		new_img[:, :, color_index] += mask * 255
		new_img[:, :, free[0]] *= neg
		new_img[:, :, free[1]] *= neg
		new_img = np.clip(new_img, 0, 255)

	return new_img, index_to_color_borders_dict


def process_video(video_in, video_out, fps=10, video_shape=(1280, 720),
                  clipping=0.5, class_area_coef=0.5,
                  db_eps=1, min_area_coef=1 / 10.):
	cap = cv2.VideoCapture(video_in)

	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	out = cv2.VideoWriter(video_out, fourcc, fps, video_shape)

	frame_count = 0
	while (cap.isOpened()):
		frame_count += 1
		ret, frame = cap.read()
		if not ret:
			break

		new_img, index_to_color_borders_dict = get_detection_img(
			frame,
			clipping=clipping,
			class_area_coef=class_area_coef,
			min_area_coef=min_area_coef)

		for class_num, color_objects in index_to_color_borders_dict.items():
			color_index, objects = color_objects
			for obj in objects:
				down, up, left, right = obj

				cv2.rectangle(new_img,
				              (right, up), (left, down),
				              color[color_index], thickness=2,
				              lineType=8, shift=0)

				cv2.putText(new_img, str(class_list[class_num][0]),
				            (left + 4, down + 20),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
				            color[color_index], 2)

		cv2.imshow('frame', new_img)
		out.write(new_img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	out.release()
	cap.release()
	cv2.destroyAllWindows()
	return frame_count
