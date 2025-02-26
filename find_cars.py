# Note: Various functions are taken or modified from Udacity's SDC course

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip

from skimage.feature import hog
import numpy as np
import pickle
import cv2
import glob
import os

DEBUG = False

IN_VIDEO = './project_video.mp4'
OUT_VIDEO = './processed_video.mp4'

CAR_DATA = ['./data/vehicles/GTI_Far',
            './data/vehicles/GTI_MiddleClose',
            './data/vehicles/GTI_Left',
            './data/vehicles/GTI_Right',
            './data/vehicles/KITTI_extracted']

NON_CAR_DATA = ['./data/non-vehicles/GTI',
                './data/non-vehicles/Extras']


def convert_rgb_to(img, conv='YCrCb'):
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, cells_per_step_x, cells_per_step_y, spatial_size, hist_bins):
    draw_img = np.copy(img)

    img = img.astype(np.float32) / 255
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_rgb_to(img_tosearch, conv='YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1 # 159
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient * cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1 # 7
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step_x + 1 # 76 + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step_y + 1 

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step_y
            xpos = xb * cells_per_step_x

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 255, 0), 2) 

                boxes.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
#            elif DEBUG:
#                xbox_left = np.int(xleft * scale)
#                ytop_draw = np.int(ytop * scale)
#                win_draw = np.int(window * scale)
#                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart), (255, 0, 0), 2) 

    if DEBUG:
        plt.imshow(draw_img)
        plt.show()

    return boxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


GREEN = (0, 255, 0)
def draw_labeled_bboxes(img, labels, is_valid):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        if is_valid(bbox):
            cv2.rectangle(img, bbox[0], bbox[1], GREEN, 2)
    # Return the image
    return img


def extract_features(img_file):
    img = mpimg.imread(img_file)
    conv_img = convert_rgb_to(img, conv='YCrCb')
    hog1_features = get_hog_features(conv_img[:,:,0], 9, 8, 2)
    hog2_features = get_hog_features(conv_img[:,:,1], 9, 8, 2)
    hog3_features = get_hog_features(conv_img[:,:,2], 9, 8, 2)
    spatial_features = bin_spatial(conv_img)
    hist_features = color_hist(conv_img)

    # Scale features and make a prediction
    return np.concatenate((spatial_features, hist_features, hog1_features, hog2_features, hog3_features))


from collections import deque
frame_detection_buffer = deque(maxlen=10)

svc = None
X_scaler = None

def process_image(image):
    global svc
    global X_scaler

    out_img = np.copy(image)

    boxes = []
    boxes += find_cars(image, 380, 500, 1.25, svc, X_scaler, 9, 8, 2, 1, 1, (32, 32), 32)
    boxes += find_cars(image, 380, 550, 1.7, svc, X_scaler, 9, 8, 2, 1, 1, (32, 32), 32)
    boxes += find_cars(image, 380, 640, 2.0, svc, X_scaler, 9, 8, 2, 2, 2, (32, 32), 32)
    boxes += find_cars(image, 380, 680, 3.0, svc, X_scaler, 9, 8, 2, 3, 1, (32, 32), 32)
    boxes += find_cars(image, 380, 680, 4.0, svc, X_scaler, 9, 8, 2, 4, 1, (32, 32), 32)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    add_heat(heat, boxes)
    if DEBUG:
        plt.imshow(heat)
        plt.show()

    # Threshold heatmap to discard pixels with less than 3 overlapping windows
    heat_thresh = apply_threshold(heat, 2)
    heat_thresh[heat_thresh > 1] = 1
    if DEBUG:
        plt.imshow(heat_thresh)
        plt.show()

    # Append to the circular frame buffer and discard pixels with less than 8 detections
    # over the last 10 frames
    frame_detection_buffer.append(heat_thresh)
    detections_heatmap = np.sum(frame_detection_buffer, axis=0)
    detections_thresh = apply_threshold(detections_heatmap, 7)
    if DEBUG:
        plt.imshow(detections_thresh, cmap='gray')
        plt.show()

    from scipy.ndimage.measurements import label
    labels = label(detections_thresh)

    if DEBUG:
        print(labels[1], 'cars found')
        plt.imshow(labels[0], cmap='gray')
        plt.show()

    # A bounding box is not valid if the H/W or W/H ratios are too large
    def is_valid_box(bbox, max_sides_ratio=3):
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]

        ratio = abs(y2 - y1) / abs(x2 - x1)
        if ratio > max_sides_ratio or 1 / ratio < 1 / max_sides_ratio:
            return False
        return True

    out_img = draw_labeled_bboxes(out_img, labels, is_valid_box)
    if DEBUG:
        plt.imshow(out_img)
        plt.show()

    return out_img


def main():
    global svc
    global X_scaler

    file_name = './svc_pickle.p'

    from pathlib import Path
    my_file = Path(file_name)
    if my_file.is_file():
        dist_pickle = pickle.load(open(file_name, 'rb'))
        svc = dist_pickle['svc']
        X_scaler = dist_pickle['scaler']

    if not svc or not X_scaler:
        print('loading training data...')
        car_image_files = []
        for folder in CAR_DATA:
            for image_file in glob.glob(os.path.join(folder, '*.png')):
                car_image_files.append(image_file)
        car_features = [extract_features(image_file) for image_file in car_image_files]
    
        non_car_image_files = []
        for folder in NON_CAR_DATA:
            for image_file in glob.glob(os.path.join(folder, '*.png')):
                non_car_image_files.append(image_file)
        non_car_features = [extract_features(image_file) for image_file in non_car_image_files]
    
        y = np.hstack((np.ones(len(car_features)),
                       np.zeros(len(non_car_features))))
    
        features = car_features + non_car_features
    
        X = np.vstack(features).astype(np.float64)
        from sklearn.preprocessing import StandardScaler
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        X = X_scaler.transform(X)
    
        rand_state = np.random.randint(0, 100)
        from sklearn.utils import shuffle
        X, y = shuffle(X, y, random_state=rand_state) 
    
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)
    
        from sklearn.svm import LinearSVC
        # Use a linear SVC (support vector classifier)
        svc = LinearSVC()
        # Train the SVC
        svc.fit(X_train, y_train)

        if DEBUG:
            print(mpimg.imread(car_image_files[0]))

            print(X_test.shape)
            pred = svc.predict(X_test[0:100])
            print('My SVC predicts:')
            print(pred)
            print('For labels:')
            print(y_test[0:100])

            print('Test Accuracy of SVC = ', svc.score(X_test, y_test))

    fileObject = open('./svc_pickle.p', 'wb')
    dist_pickle = {'svc': svc,
                   'scaler': X_scaler}
    pickle.dump(dist_pickle, fileObject)
    fileObject.close()

    clip1 = VideoFileClip(IN_VIDEO).subclip(20, 45)
    out_clip = clip1.fl_image(process_image)
    out_clip.write_videofile(OUT_VIDEO, audio=False)


if __name__ == '__main__':
    main()
