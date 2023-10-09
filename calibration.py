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



def calibrate_camera(H, W, red_fac, images):
    #termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #prepare object points
    objp = np.zeros((H*W, 3), np.float32)
    objp[:, :2] = np.mgrid[0:H, 0:W].T.reshape(-1, 2)

    #arrays to store object points and image points from all images
    objpoints = [] #3D points in real-world space
    imgpoints = [] #2D points in image plane

    for fname in images:
        img = cv2.imread(fname)
        img = DownSampleImage(img, red_fac)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #find chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (H, W), None)

        #if found, add object points, image points (after refining)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            #draw and display corners
            cv2.drawChessboardCorners(img, (H, W), corners2, ret)
            cv2.imshow("image", cv2.resize(img, (920, 540)))
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    #calibration
    ret, CameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    #save calibration results
    pickle.dump((CameraMatrix, dist), open("calibration.pkl", "wb"))

    #reprojection error
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], CameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    return mean_error/len(objpoints)


def load_calibration(file_name):
    with open(file_name, "rb") as file:
        CameraMatrix, dist = pickle.load(file)

    return CameraMatrix, dist


def undistort_image(img_in, CameraMatrix, dist):
    h, w = img_in.shape[:2]

    NewCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(CameraMatrix, dist, (w,h), 1, (w,h))
    img_out = cv2.undistort(img_in, CameraMatrix, dist, None, NewCameraMatrix)

    x, y, w, h = roi
    img_out = img_out[y:y+h, x:x+w]

    return img_out


def test_undistort(red_fac):
    img1 = cv2.imread("images/sfm2/20231008_230707.jpg")
    img1 = DownSampleImage(img1, red_fac)

    CameraMatrix, dist = load_calibration("calibration.pkl")
    img2 = undistort_image(img1, CameraMatrix, dist)

    cv2.imwrite("calibrated.png", img2)


if __name__ == "__main__":
    H, W = 8, 6
    red_fac = 2
    images = glob.glob("images/calibration/*.jpg")

    reproj_error = calibrate_camera(H=H, W=W, red_fac=red_fac, images=images)

    print("reprojection error: {}\n".format(reproj_error))

    test_undistort(red_fac)
