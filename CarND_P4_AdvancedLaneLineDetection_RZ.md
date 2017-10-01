
# Self-Driving Car Engineer Nanodegree


## Project 4: **Advanced Lane Finding** 
***
This project focuses on finding lane lines and lane line curvature in a video stream using the lessons learned in Udacity Self Driving Nano-Degree Program. 

Major steps to develop the lane line detection algorithm include:
- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it! :)


## Calibrating the Camera
#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the cells #2 and #3 of the IPython notebook located in "CarND_P4_AdvancedLaneLineDetection_RZ.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![png](output_7_1.png)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

Color and gradinet thresholding functions are defined in cels #5 to #9 of the notebook to generate the binary image with clearly visible lane lines. 

Results of each transformation are illustrated in below:

![png](output_31_0.png)



![png](output_31_1.png)



![png](output_31_2.png)



![png](output_31_3.png)



![png](output_31_4.png)



![png](output_31_5.png)



![png](output_31_6.png)



![png](output_31_7.png)



![png](output_31_8.png)



![png](output_31_9.png)



![png](output_31_10.png)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective Transformation is performed in cels #11 and #12 of the notebook. First, source and destination points are found using an interactive image with straight lines. I chose the hardcode the source and destination points in the following manner:


This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

The points are then used to perfroma the perspective transformation on an image with curved lines. Both images are illustrated below.

![png](output_18_1.png)



![png](output_18_2.png)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Sliding window method as explained in the course is used to identify lane lines and fit polynomials. The code is implemented in the function "find_lanes" in cell #47. Unwarped images are fed to this function. Window size is set to 9, minimum number of pixles is set to 30 and the margin for the width of the window is set to 70 pixles. For the first frame, the histogram is used to find the bottom window. For the frames after that, the previous position us used in the function as a basis. A second order polynomial is used to fit the lane lines as illustrated in the image below:


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cell #17 of the notebook. Conversion between pixles and meters is performed as:

```python
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

Then polynomials are fit to the left and right curves as explained in the course and coded in cell #17.
Distance to the center of the road is implemented in cell # 18.



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the "pipeline' function in cell #19.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion
#### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?
Since the code is developed using the provided images and the video which have specific lightning conditions, the code might experience difficulties for extremely diffetent situations in terms of lighting and colors. Also, lane exits, merges and crossroads might be challenging to detect. If another car in front of the host car blocks the camera, detecting the lane lines willl be difficult. 

To improve the code, it might be possible to:
- Estimate the optimized thresholds for the threshold parameters
- Add other color spaces 
- Estimate the source and destination points usnig multiple images 

