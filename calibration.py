import cv2
import glob
import numpy as np
import pickle


def DownSampleImage(image, reduction_factor):
    """
    Downsample image reduction_factor number of times
    """
    for _ in range(reduction_factor):
        row = image.shape[0]
        col = image.shape[1]

        image = cv2.pyrDown(image, dstsize = (col//2, row//2))
    return image


class Calibration():
    def __init__(self, intrinsic=None, dist=None,
                 size=None, reduction_factor=0):
        self.OrigCameraMatrix = intrinsic
        self.DisParams = dist
        self.OrigSize = size
        self.RedFac = reduction_factor
        self.CameraMatrix = None
        self.Crop = None
        self.Size = None
        if (intrinsic is not None) and (dist is not None) and (size is not None):
            self.CameraMatrix, roi = self._opt_camera_matrix(intrinsic, dist, size)
            self.Crop = roi[:2]
            self.Size = roi[:-3:-1]

    @staticmethod
    def _opt_camera_matrix(intrinsic, dist, size):
        h, w = size
        CameraMatrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic, dist, (w,h), 1, (w,h))

        return CameraMatrix, roi

    def load(self, file_name):
        with open(file_name, "rb") as file:
            h, w, rf, intrinsic, dist = pickle.load(file)

        self.OrigCameraMatrix = intrinsic
        self.DisParams = dist
        self.OrigSize = (h, w)
        self.RedFac = rf

        self.CameraMatrix, roi = self._opt_camera_matrix(intrinsic, dist, (h, w))
        self.Crop = roi[:2]
        self.Size = roi[:-3:-1]

    def save(self, file_name):
        with open(file_name, "wb") as file:
            pickle.dump((self.OrigSize[0], self.OrigSize[1], self.RedFac,
                         self.CameraMatrix, self.DisParams), file)

    def undistort(self, img_in):
        h, w = img_in.shape[:2]
        if h!=self.OrigSize[0] and w!=self.OrigSize[1]:
            raise SystemExit('size mismatch between input image and calibration images')

        img_out = cv2.undistort(img_in, self.OrigCameraMatrix, self.DisParams,
                                None, self.CameraMatrix)

        x, y = self.Crop
        h, w = self.Size
        img_out = img_out[y:y+h, x:x+w]

        return img_out


def calibrate_camera(board_size, red_fac, images):
    #termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #prepare object points
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    #arrays to store object points and image points from all images
    objpoints = [] #3D points in real-world space
    imgpoints = [] #2D points in image plane

    for fname in images:
        img = cv2.imread(fname)
        img = DownSampleImage(img, red_fac)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #find chess board corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        #if found, add object points, image points (after refining)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            #draw and display corners
            cv2.drawChessboardCorners(img, board_size, corners2, ret)
            cv2.imshow("image", cv2.resize(img, (920, 540)))
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if not objpoints:
        print("Calibration failed. Try different inputs.")
        return

    #calibration
    ret, CameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    CamCal = Calibration(intrinsic=CameraMatrix,
                         dist=dist,
                         size=img.shape[:2],
                         reduction_factor=red_fac)

    #reprojection error
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], CameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    mean_error/=len(objpoints)

    print("Calibration successful")
    print(f"Reprojection error: {mean_error}")

    return CamCal


def test_undistort():
    CamCal = Calibration()
    CamCal.load("calib.pkl")

    img1 = cv2.imread("images/sfm_test/20231008_230707.jpg")
    img1 = DownSampleImage(img1, CamCal.RedFac)

    img2 = CamCal.undistort(img1)

    #cv2.imwrite("calibrated.png", img2)
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img1[:,:,::-1])
    ax[0].set_title("raw image")
    ax[1].imshow(img2[:,:,::-1])
    ax[1].set_title("undistorted image")
    plt.show()


if __name__ == "__main__":
    H, W = 8, 6
    red_fac = 2
    images = glob.glob("images/calibration/*.jpg")

    CamCal = calibrate_camera(board_size=(H, W), red_fac=red_fac, images=images)

    CamCal.save("calib.pkl")

    test_undistort()
