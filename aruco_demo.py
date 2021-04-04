import pickle as pkl
import numpy as np
import cv2,os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.transform import Rotation as R
from cv2 import aruco

class Aruco:
    def __init__(self):
        self.mtx=np.load('aruco_dir/datacalib_mtx_webcam.pkl', allow_pickle=True)
        self.dist=np.load('aruco_dir/datacalib_dist_webcam.pkl', allow_pickle=True)
        self.size_of_marker =  0.0145
        self.length_of_axis = 0.01
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.parameters =  aruco.DetectorParameters_create()
        print("Aruco object created")
        
    def aruco(self,frame):
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, self.aruco_dict, parameters=self.parameters)
        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        rvecs,tvecs,trash = aruco.estimatePoseSingleMarkers(corners, self.size_of_marker , self.mtx, self.dist)
        imaxis = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        if tvecs is not None:
            for i in range(len(tvecs)):
                imaxis = aruco.drawAxis(imaxis, self.mtx, self.dist, rvecs[i], tvecs[i], self.length_of_axis)
                rvec=np.squeeze(rvecs[0], axis=None)
                tvec=np.squeeze(tvecs[0], axis=None)
                tvec=np.expand_dims(tvec, axis=1)
                rvec_matrix = cv2.Rodrigues(rvec)[0]
                proj_matrix = np.hstack((rvec_matrix,tvec))
                euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                cv2.putText(imaxis, 'X: '+str(int(euler_angles[0])),(10, 30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))
                cv2.putText(imaxis, 'Y: '+str(int(euler_angles[1])),(115, 30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
                cv2.putText(imaxis, 'Z: '+str(int(euler_angles[2])),(200, 30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0))
        cv2.imshow('Aruco',imaxis)
                
