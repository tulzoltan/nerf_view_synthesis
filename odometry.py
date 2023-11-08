import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt

from calibration import Calibration, ImageLoader


class VisualOdometry():
    def __init__(self, CamCal, images, display_matches=False):
        self.dismat = display_matches
        self.CamCal = CamCal
        self.images = images
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and 
        translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i: int):
        """
        This function detects keypoints and descriptors from 
        the i-1'th and i'th image using the orb object

        Parameters
        ----------
        i (int): index of current frame

        Returns
        -------
        q1 (ndarray): good keypoint matches in i-1'th image
        q2 (ndarray): good keypoint matches in i'th image
        """
        #Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(self.images[i-1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i]  , None)
        #Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        #Find the matches that do not have a large distances
        #Store all the good matches as per Lowe's ratio test
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        if self.dismat:
            draw_params = dict(matchColor = -1, #draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, #draw only inliers
                 flags = 2)

            img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1], 
                               kp2, good ,None,**draw_params)
            cv2.imshow("image", img3)
            cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): good keypoint matches in i-1'th image
        q2 (ndarray): good keypoint matches in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.CamCal.CameraMatrix, threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): good keypoint matches in i-1'th image
        q2 (ndarray): good keypoint matches in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            #Get the transformation matrix
            T = self._form_transf(R, t)
            #Make the projection matrix
            ProMat = np.concatenate((self.CamCal.CameraMatrix, np.zeros((3, 1))), axis=1)
            P = ProMat @ T

            #Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(ProMat, P, q1.T, q2.T)
            #Also seen from cam 2
            hom_Q2 = T @ hom_Q1

            #Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            #Find the number of points there has positive z coordinate 
            #in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        #Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        #Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        #Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        #Select the pair that has the most points with positive 
        #z coordinates
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]


def estimate_egomotion(CamCal: Calibration,
                       images,
                       display_matches: bool = False):
    #get egomotion data
    vo = VisualOdometry(CamCal=CamCal,
                        images=images,
                        display_matches=display_matches)

    trajectory = []
    for i in tqdm(range(len(vo.images))):
        if i == 0:
            cur_pose = np.eye(4)
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = cur_pose @ np.linalg.inv(transf)
        trajectory.append(cur_pose)

    if display_matches:
        cv2.destroyAllWindows()

    return trajectory


if __name__ == "__main__":
    #load calibration data
    CamCal = Calibration()
    CamCal.load("calib.pkl")

    #get images
    dataset = ImageLoader(directory="images/sfm_test",
                          CamCal=CamCal,
                          grayscale=True)

    est_path = estimate_egomotion(CamCal=CamCal,
                                  images=dataset,
                                  display_matches=True)

    est_path = np.array(est_path)

    #plot path
    fig = plt.axes(projection='3d')
    fig.plot3D(est_path[:, 1, 3], est_path[:, 0, 3], est_path[:, 2, 3])
    plt.show()
