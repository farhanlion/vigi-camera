from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import xml.etree.ElementTree as ET
import argparse
import imutils
import cv2
import pybgs as bgs
import time
import numpy as np
import datetime
from subsense import SuBSENSE
import os
from pathlib import Path
from uploader import Uploader

class Surveillance:
	def __init__(self):
		self.s = 0
		self.vs = WebcamVideoStream(src=0).start()
		self.fps = FPS().start()
		self.Subsense = SuBSENSE()
		self.uploader = Uploader()

		self.mode= 'light' # for nightmode
		self.Subsense.switchMode(self.mode)
		self.width = 0
		self.height = 0
		self.size = 0
		
		# movements
		self.firstmovment = True
		self.firstmovmenttime = 0
		self.num_movements = 0
		self.old_num_movements = 0 
		self.all_movements = []

		# record
		self.recording = False
		self.recordedframes = []
		self.num_recorded_frames=0
		self.currtimestamp = 0
		self.videonumber = 1
		

	def checkDarkness(self, frame):
		meanpercent = np.mean(frame) * 100 / 255
		darkness = "dark" if meanpercent < 30 else "light"
		# print(f'{classification} ({meanpercent:.1f}%)')
		return darkness

	def changeMode(self, mode):
		self.mode = mode
		self.Subsense.switchMode(mode)

	def nightModeFilter(self, frame, contrast=5.0, brightness=1.0):
		normalised_frame = cv2.normalize(frame, None, 0,255,cv2.NORM_MINMAX)
		night_frame = cv2.addWeighted( frame, contrast, frame, 0.1, brightness)
		night_frame = cv2.fastNlMeansDenoising(night_frame,None,10, 7, 21)
		night_frame = cv2.fastNlMeansDenoising(night_frame,None,10, 7, 21)
		return night_frame, normalised_frame
	
	def resetBackground(self):
		for i in range(10):
			frame = self.vs.read()
			frame = cv2.resize(frame, (640, 360))
			new_frame = self.Subsense.apply(frame)
		return new_frame
	
	def checkLocation(self,x,y):
		if x > (self.width/2) and y > (self.height/2):
			location = 'bottom right'
			return location
		elif x < (self.width/2) and y > (self.height/2):
			location = 'bottom left'
			return location
		elif x > (self.width/2) and y < (self.height/2):
			location = 'top right'
			return location
		elif x < (self.width/2) and y < (self.height/2):
			location = 'top left'
			return location

	def checkSize(self, contour):
		if cv2.contourArea(contour)<15000:
			size = 'small'
		else:
			size = 'big'
		return size
	
	def drawContours(self,frame, fgmask):
		# find the contours around detected objects
		contours, hierarchy = cv2.findContours(fgmask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		self.num_movements = 0
		for contour in contours:
			movement = {}
			if cv2.contourArea(contour) < 7000 :
				continue
			(x, y, w, h) = cv2.boundingRect(contour)
			self.num_movements+=1

			if self.num_movements>self.old_num_movements:
				centerx = x+w/2
				centery = y+h/2
				location = self.checkLocation(centerx,centery)
				object_size = self.checkSize(contour)

				self.old_num_movements=self.num_movements
				movement['location'] = location
				movement['object_size'] = object_size
				self.all_movements.append(movement)

			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
		
		return frame
	
	def startRecording(self):
		self.currtimestamp = datetime.datetime.now()
		self.recording = True
		self.num_recorded_frames=0
		print("recording")
	
	def recordFrame(self, frame):
		self.num_recorded_frames += 1
		self.recordedframes.append(frame)

	def stopRecording(self):
		# print(self.num_movements)
		self.recording = False

	def saveFramesToVideo(self):

		print('writing')
		print(len(self.recordedframes), ' Frames')

		path = "./videos/movement"+str(self.videonumber)+".mp4"

		out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'), 10, self.size)	
		for frame in self.recordedframes:
			out.write(frame)# writing to a image array
		
		out.release()
		print('done')
		self.videonumber +=1
		self.recordedframes = []
		return path

	def run(self):
		frame = self.vs.read()
		frame = cv2.resize(frame, (640, 360))

		self.height, self.width, layers = frame.shape
		self.size = (self.width,self.height)

		night_frame = frame
		frame = self.resetBackground()
		
		while True:
			
			keyboard = cv2.waitKey(1)
			if keyboard == 27:
				break
			
			# read frame
			frame = self.vs.read()
			frame = cv2.resize(frame, (640, 360))

			# check how dark it is (returns 'dark' or 'light')
			darkness = self.checkDarkness(frame)

			# if changed mode
			if darkness != self.mode:
				self.changeMode(darkness) 
				cv2.destroyAllWindows()
				return True
			
			# apply dark filter for better foreground detection, show respective frames
			self.num_movements = 0
			if self.mode == 'dark':
				night_frame, normaized_frame = self.nightModeFilter(frame)
				fgmask = self.Subsense.apply(night_frame)
				frame = self.drawContours(normaized_frame, fgmask)
			else: 
				fgmask = self.Subsense.apply(frame)
				frame = self.drawContours(frame, fgmask)



			if self.num_movements>0 and self.recording == False:
				self.startRecording()
			
			if self.recording and self.num_recorded_frames<=50:
				self.recordFrame(frame)
				cv2.imshow("Recorded frame", frame)

				if self.num_recorded_frames==50:
					self.stopRecording()
					savedvideo = self.saveFramesToVideo()
					print('saved video', savedvideo)
					#self.uploader.upload(savedvideo, self.currtimestamp,self.all_movements)


			cv2.imshow("mask", fgmask)
			cv2.imshow("Frame",frame)
			
			self.fps.update()

		self.fps.stop()
		print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

		cv2.destroyAllWindows()
		self.vs.stop()
		return False


if __name__ == "__main__":
	s = True
	app = Surveillance()
	while s:
		s = app.run()
		app.Subsense = SuBSENSE()