#!/usr/bin/env python3

import cv2
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
	"""Draw a collection of lines on an image.
	"""
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
	cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)

def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
	h, w = shape
	y_min, x_min, y_max, x_max = bbox
	p1 = (int(y_min * h), int(x_min * w))
	p2 = (int(y_max * h), int(x_max * w))
	cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
	p1 = (p1[0] + 15, p1[1])
	cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
	h, w = img.shape
	for i in range(bboxes.shape[0]):
		y_min, x_min, y_max, x_max = bboxes[i]
		color = colors[classes[i]]

		# Draw bounding box
		p1 = (int(y_min * h), int(x_min * w))
		p2 = (int(y_max * h), int(x_max * w))
		cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)

		# Draw text
		text = '%s/%.3f' % (classes[i], scores[i])
		p1 = (p1[0] - 5, p1[1])
		cv2.putText(img, text, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)
	

