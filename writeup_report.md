## Writeup For Vehicle Detection Project
### A linear SVM classifier has been trained with Histogram of Oreiented Gradients (HOG) features and binned color features, which are extracted from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself. The Trained SVM classifier detects vehicles well in video project_video.mp4 and meet project primary requirments.

### Multi-scale windows and heat threshold between diffient frames have been used in this pipeline to improve performance.

### As Tips and Tricks for the Project mentioned, it is very important to make clear what kind of image type you take in and whether necessary to normalize.

### Thanks a lot for these great ideas from below links:
[Vehicle Detection and Tracking using Computer Vision, written by Arnaldo Gunzi](https://chatbotslife.com/vehicle-detection-and-tracking-using-computer-vision-baea4df65906#.808k7t2tv)

[Vehicle tracking using a support vector machine vs. YOLO, written by Kaspar Sakmann](https://medium.com/@ksakmann/vehicle-detection-and-tracking-using-hog-features-svm-vs-yolo-73e1ccb35866?spm=5176.100239.blogcont71662.16.XxC6bS#.8s1jsyx7f)

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./p5_test_video.mp4
[video2]: ./p5_project_video.mp4
[video3]: ./p5_out_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
More details you can review jupyter notebook file.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook.

`#data exploration and scikit-image HOG features`

I started by reading in all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` to get a feel for what the `skimage.hog()` output looks like:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally settled HOG parameters as following:

```
### Parameters for features
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
```

The finnally HOG parameters' choice are based on:

1. these parameters are used in class
2. the comparison results from Arnaldo Gunzi in his post [Vehicle Detection and Tracking using Computer Vision](https://chatbotslife.com/vehicle-detection-and-tracking-using-computer-vision-baea4df65906#.808k7t2tv)
3. the most and important reason is that I tested and confirmed that these parameters performs well in test images shown below and with consideration of calculation time cost.

An important issue needs more attention is that the predict acccuracy interrelates with cars detection performance but not directly. In my mind, it is not necessary to spend too much time to optimize these parameters and should verify in vedio detection simulation.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using these following codes:

```
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
```

Before trainning, the train and test data should be generated, normalized and random shuffled.
```
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
```
After sucessful trainning, I saved trained SVM classifer into `svc_pickle.p`. It is more convinient to load it for prediction without training.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Similar to multi-scale windows in class, the below example image with different detecting zones is shown below. The target is to use different scales to make sure more accurate detection with less time cost.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on only one scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

In the finnally detecting pipeline, several scales are used together with color features.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's:
1. [test_video.mp4 result](./p5_test_video.mp4)
2. [project_video.mp4 result](./p5_project_video.mp4)
3. [project_video.mp4 result with advanced line detection ](./p5_out_project_video.mp4)

The last video is processed by classifier based on the previous project results. This agree with engineering modular thinking and step by step.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. uI constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 1 frame corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from different scales:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As Tips and Tricks for the Project mentioned, it is very important to make clear what kind of image type you take in and whether necessary to normalize.

I spent a lot of time to debug why the classifier does not work, finally found that the jpg format images should be normalized first before feed them to train.

Class Vehicle() is established here to save every heat from frames and then everage 10 frames to show. In this case, `apply_threshold()` funtion is used twice. The first time is to filter different scales and the second time is to filter different frames.

In the finally results video, there are two 'false positive' for the opposite cars. It is more distinct with hollow crawl.

Whether should we detect the opposite cars and how to avoid it? This is a very interesting topic.
