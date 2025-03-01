import numpy as np
import cv2 as cv
import glob
import argparse

# Argument parser
# parser = argparse.ArgumentParser(description="Lucas-Kanade Optical Flow with images from a directory")
# parser.add_argument('directory', type=str, help='Path to the image directory')
# args = parser.parse_args()

directory = "/home/kewei/rain_data/oxford/10-29/left_undistort"
# directory = "/home/kewei/rain_data1/"
# directory = "/home/kewei/ACVNet/data/oxford/left"
# Get sorted list of image paths
# image_paths = sorted(glob.glob(f"{args.directory}/*.[jp][pn]g"))  # Matches .jpg, .jpeg, .png
image_paths = sorted(glob.glob(f"{directory}/*.[jp][pn]g"))  # Matches .jpg, .jpeg, .png
if not image_paths:
    print("No images found in the directory.")
    exit()

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Load the first frame
old_frame = cv.imread(image_paths[0])
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Process all images
for img_path in image_paths[1:]:  # Start from the second image
    frame = cv.imread(img_path)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if p0 is None:
        print("No good features to track!", img_path)
        continue
    # Calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        img = cv.add(frame, mask)
        cv.imshow('Optical Flow', img)

        # Press ESC to exit
        key = cv.waitKey(0) & 0xFF
        if cv.waitKey(30) & 0xFF == 27:
            break

        # Update the previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        print("No good features found in this frame.")

cv.destroyAllWindows()



#### it works if directory = "/home/kewei/rain_data1/", it couldn't work if directory = directory = "/home/kewei/rain_data/oxford/10-29/left_undistort"
## because there's no good feature to capture in the first image. P0 = None. and it wasn't updated but trap in the for loop.