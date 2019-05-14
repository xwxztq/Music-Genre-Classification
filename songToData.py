# -*- coding: utf-8 -*-
from subprocess import Popen, PIPE, STDOUT
import os
from PIL import Image
import eyed3

from sliceSpectrogram import createSlicesFromSpectrograms, sp2slice
from audioFilesTools import isMono, getGenre
from config import rawDataPath
from config import spectrogramsPath
from config import pixelPerSecond

#Tweakable parameters
desiredSize = 128

#Define
currentPath = os.path.dirname(os.path.realpath(__file__)) 

#Remove logs
eyed3.log.setLevel("ERROR")

#Create spectrogram from mp3 files
def createSpectrogram(filename,newFilename):
	#Create temporary mono track if needed
	print("This is the new filename",newFilename)
	if isMono(rawDataPath+filename):
		command = "cp '{}' './tmp/{}.mp3'".format(rawDataPath+filename,newFilename)
	else:
		command = "sox '{}' './tmp/{}.mp3' remix 1,2".format(rawDataPath+filename,newFilename)
	
	print("This is the first command",command)
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
	print("Open successfully")
	output, errors = p.communicate()
	if errors:
		print(errors)
	#Create spectrogram
	filename.replace(".mp3","")
	command = "sox './tmp/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(newFilename,pixelPerSecond,spectrogramsPath+newFilename)
	print("This is command",command)
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
	output, errors = p.communicate()
	print("shit")
	if errors:
		print(errors)

	#Remove tmp mono track
        
	#os.remove("./tmp/{}.mp3".format(newFilename))
def mp2png(filename,newFilename):
	#Create temporary mono track if needed
	print('wewe')
	if isMono(filename):
		command = "cp '{}' './tmp/{}.mp3'".format(filename,newFilename)
	else:
		command = "sox '{}' './tmp/{}.mp3' remix 1,2".format(filename,newFilename)
	os.system(command)

	print('fefe')
	#Create spectrogram
	filename.replace(".mp3","")
	command = "sox './tmp/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(newFilename,pixelPerSecond,newFilename)
	os.system(command)
	
	#Remove tmp mono track
        
	#os.remove("./tmp/{}.mp3".format(newFilename))

#Creates .png whole spectrograms from mp3 files
def createSpectrogramsFromAudio():
	genresID = dict()
	files = os.listdir(rawDataPath)
	files = [file for file in files if file.endswith(".mp3")]
	nbFiles = len(files)
	print(rawDataPath)
	#Create path if not existing
	if not os.path.exists(os.path.dirname(spectrogramsPath)):
		try:
			os.makedirs(os.path.dirname(spectrogramsPath))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	print(spectrogramsPath)
	#Rename files according to genre
	for index,filename in enumerate(files):
		print("Creating spectrogram for file {}/{}...".format(index+1,nbFiles))
		for i in range(len(filename)):
			if (filename[i]=='_'):
				fileGenre = filename[0:i]
		print(fileGenre)
		#fileGenre = getGenre(rawDataPath+filename)
		genresID[fileGenre] = genresID[fileGenre] + 1 if fileGenre in genresID else 1
		fileID = genresID[fileGenre]
		newFilename = fileGenre+"_"+str(fileID)
		createSpectrogram(filename,newFilename)

#Whole pipeline .mp3 -> .png slices
def createSlicesFromAudio():
	print("Creating spectrograms...")
	createSpectrogramsFromAudio()
	print("Spectrograms created!")
	print("Creating slices...")
	createSlicesFromSpectrograms(desiredSize)
	print("DesiredSize")
	print(desiredSize)
	print("Slices created!")
def mp3topng(filename):
	print("Start mp3 -> png!!")
	mp2png(filename, "new")
	print("Start png -> pngs!!")
	sp2slice("new.png", desiredSize)
	print("Finished")


