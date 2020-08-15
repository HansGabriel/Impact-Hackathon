import cv2
import threading
import time
import logging
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

logger = logging.getLogger(__name__)

thread = None

class Camera:
	def __init__(self,fps=20,video_path=0):	
		logger.info(f"Initializing camera class with {fps} fps and video_path={video_path}")
		self.fps = fps
		self.video_path = video_path
		self.vid = cv2.VideoCapture(self.video_path)
		# We want a max of 5s history to be stored, thats 5s*fps
		self.max_frames = 5*self.fps
		self.frames = []
		self.isrunning = False

		# while True:
		# 	return_value, frame = self.vid.read()
		# 	if return_value:
		# 		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# 		image = Image.fromarray(frame)
		# 	else:
		# 		print('Video has ended or failed, try a different video format!')
		# 		break
		
		# 	frame_size = frame.shape[:2]
		# 	image_data = cv2.resize(frame, (input_size, input_size))
		# 	image_data = image_data / 255.
		# 	image_data = image_data[np.newaxis, ...].astype(np.float32)
		# 	start_time = time.time()

		# 	batch_data = tf.constant(image_data)
		# 	pred_bbox = infer(batch_data)
		# 	for key, value in pred_bbox.items():
		# 		boxes = value[:, :, 0:4]
		# 		pred_conf = value[:, :, 4:]

		# 	boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
		# 		boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
		# 		scores=tf.reshape(
		# 			pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
		# 		max_output_size_per_class=50,
		# 		max_total_size=50,
		# 		iou_threshold=self.FLAGS.iou,
		# 		score_threshold=self.FLAGS.score
		# 	)

		# 	# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
		# 	original_h, original_w, _ = frame.shape
		# 	bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

		# 	pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

		# 	image = utils.draw_bbox(frame, pred_bbox, self.FLAGS.info)
			
		# 	fps = 1.0 / (time.time() - start_time)
		# 	print("FPS: %.2f" % fps)
		# 	result = np.asarray(image)
		# 	cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
		# 	result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			
		# 	if not self.FLAGS.dont_show:
		# 		cv2.imshow("result", result)
			
		# 	if self.FLAGS.output:
		# 		out.write(result)
		# 	if cv2.waitKey(1) & 0xFF == ord('q'): break
		# cv2.destroyAllWindows()

	class FLAGS: 
		def __init__(self):
			self.framework = 'tf'
			self.weights = './checkpoints/gun-416'
			self.size = 416
			self.tiny = False
			self.model = 'yolov4'
			self.video = '0'
			self.output = None
			self.output_format = 'XVID'
			self.iou = 0.45
			self.score = 0.25
			self.count = False
			self.dont_show = False
			self.info = False
		
	def run(self):
		logging.debug("Perparing thread")
		global thread
		if thread is None:
			logging.debug("Creating thread")
			thread = threading.Thread(target=self._capture_loop,daemon=True)
			logger.debug("Starting thread")
			self.isrunning = True
			thread.start()
			logger.info("Thread started")

	def _capture_loop(self):
		dt = 1/self.fps
		self.FLAGS = self.FLAGS()
		config = ConfigProto()
		config.gpu_options.allow_growth = True
		session = InteractiveSession(config=config)
		STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(self.FLAGS)
		input_size = self.FLAGS.size

		saved_model_loaded = tf.saved_model.load(self.FLAGS.weights, tags=[tag_constants.SERVING])
		infer = saved_model_loaded.signatures['serving_default']
		logger.debug("Observation started")

		out = None

		while self.isrunning:
			return_value, frame = self.vid.read()
			if return_value:
				if len(self.frames)==self.max_frames:
					self.frames = self.frames[1:]
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(frame)

				frame_size = frame.shape[:2]
				image_data = cv2.resize(frame, (input_size, input_size))
				image_data = image_data / 255.
				image_data = image_data[np.newaxis, ...].astype(np.float32)
				start_time = time.time()

				batch_data = tf.constant(image_data)
				pred_bbox = infer(batch_data)
				for key, value in pred_bbox.items():
					boxes = value[:, :, 0:4]
					pred_conf = value[:, :, 4:]

				boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
					boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
					scores=tf.reshape(
						pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
					max_output_size_per_class=50,
					max_total_size=50,
					iou_threshold=self.FLAGS.iou,
					score_threshold=self.FLAGS.score
				)

				# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
				original_h, original_w, _ = frame.shape
				bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

				pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

				image = utils.draw_bbox(frame, pred_bbox, self.FLAGS.info)
				
				self.frames.append(frame)
			time.sleep(dt)
		logger.info("Thread stopped successfully")

	def stop(self):
		logger.debug("Stopping thread")
		self.isrunning = False
	def get_frame(self, _bytes=True):
		if len(self.frames)>0:
			if _bytes:
				img = cv2.imencode('.png',self.frames[-1])[1].tobytes()
			else:
				img = self.frames[-1]
		else:
			with open("images/not_found.jpeg","rb") as f:
				img = f.read()
		return img
		