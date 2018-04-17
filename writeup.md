## Writeup Template

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/compare_non_vehicle_1.jpg
[image2]: ./output_images/compare_non_vehicle_2.jpg
[image3]: ./output_images/compare_hog_vehicle.jpg
[image4]: ./output_images/compare_hog_non_vehicle.jpg
[image5]: ./output_images/compare_bin_vehicle.jpg
[image6]: ./output_images/compare_bin_non_vehicle.jpg
[image7]: ./output_images/compare_historgram_vehicle.jpg
[image8]: ./output_images/compare_historgram_non_vehicle.jpg
[image9]: ./output_images/train_model_linear.jpg
[image10]: ./output_images/train_model_rbf.jpg
[image11]: ./output_images/compare_windows_1.jpg
[image12]: ./output_images/compare_windows_1_5.jpg
[image13]: ./output_images/compare_windows_2.jpg
[image14]: ./output_images/compare_windows_2_5.jpg
[image15]: ./output_images/compare_windows_4.jpg
[image16]: ./output_images/compare_windows.jpg
[image17]: ./output_images/compare_heatmap.jpg
[image18]: ./output_images/compare_pipeline.jpg
[gif1]: ./output_images/output_gif.gif

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)
#### 0. Previewing the data
To see how this previews are generated take a look at [task preview data.py](src/task_preview_data.py).

Before starting I wanted to preview the data.  
We happen to have **8792** images of vehicles and **8968** images of non-vehicles, that's only 176 more images so we can say that the samples are pretty balanced.  
Images have a dimension of 64x64x3, this is important because our classifier will learn to find cars in 64x64 images and thus we'll need to feed it images of that size when performing the window search.


Here are some examples comparing vehicle and non-vehicle images, most of the time the difference is pretty clear for a human (so let's see how our algorythms perform).
 
![alt text][image1]
![alt text][image2]

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
To see how this previews are generated take a look at [task hog data.py](src/task_hog_data.py).

Hog features are extacted using the `hog` method from the `skimage` library ([prepare data data.py](src/tools/prepare_data.py))

To train my final model I used the **YCrCb** colorspace with the following parameters:

- orient: 18
- pix per cell: 8
- cell per block: 2

Here's an example of how the hog features look with those settings:

For a vehicle
![alt text][image3]

For a non-vehicle
![alt text][image4]


#### 2. Explain how you settled on your final choice of HOG parameters.

Before having the model, I performed some hog previews in different color spaces, and with different parameters. 

Initially I decided to use a combination color spaces and features that I felt that extracted the clearer features (something on the lines of combining HSV+YCrCb and using a lot of orients & pix per cell), but after training the model those didn't seem to perform very well and was terribly slow, so I decided to take a different approach and try to find what color space and parameters worked better for the classifier and had a good balance between accuracy and number of params by trial & error (this is important as a bigger model will predict more slowly).

Finally I found out that using the parameters explained above worked the best.

#### 2.1. Other features: spatial bins and color histograms.

Appart from the HOG, I decided to also add some extra features, specifically I added **16x16 spatal bins** (preview generated in [task spatial bin data.py](src/task_spatial_bin_data.py).) 
 
![alt text][image5]
![alt text][image6]

And also added **68 bin histograms** on the 3 channels of the YCrCb image (preview generated in [task historgram bin data.py](src/task_histogram_data.py).)

![alt text][image7]
![alt text][image8]

The methods used for generating those features can be found in [prepare data data.py](src/tools/prepare_data.py) 

#### 2.2. Combining features

I finally flattened and combined all those features, generating a 11556 features vector (for each image).

The methods for generating those features is `combined_features` in [prepare data data.py](src/tools/prepare_data.py) 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

**Loading the data**

The code where I train the model can be found in [task model train.py](src/task_model_train.py).

First of all I loaded all the images, both vehicle and non-vehicle, and proceeded to the feature extraction. This process is slow (takes around 2.5 minutes).

In order to separate train and validation splits, my first attempt was to use just shuffle the data and then take 20% of both vehicle and non-vehicle images for the validation.  
After training my model the validation accuracy was pretty high, getting to almost 99% but actually the model was overfitting the data, because the images in the train and validation splits where too similar. When I tried to use this model in the video the performance was really bad, the model learned to recognize only images from the dataset.

Because the vehicle images are generated based on time series, I decided to just pick the last 20% of them before shuffling, the validation accuracy dropped to around 94% but the performance of the model in the video was a lot better.

**Normalizing the data**

To normalize the data I used a StandardScaler but just fitted it on the training split (to avoid overfitting the validation data). I then used the same scaler when generating predictions

**Trainign the model**

I trained a `LinearSVC` **with C=0.1** (I tried a regular SVC with likear kernel but the training and prediction times were much higher).

After training, I saved the scaled and model to a pickle file called [svc linear.p](src/svc_linear.p) 

Here's the output after training the model:
![alt text][image9]

A prediction time of 0.5ms is very good, because we will need to perform predictions on 729 windows, so the prediction time alone for a single frame will be of about 390ms.
The feature extraction time is very high, 10ms per image, for 729 windows it would take 7.3s but luckily we'll perform some optimization to reduce this time.

**Other models**

I tried a lot of combination for parameters, the linear model above gave a pretty good balance of accuracy and prediction time.
I also wanted to see how a model using **rbf** (C=10) performed, so I trained one [svc lrbf.p](src/svc_rbf.p) I had to reduce some features so the training wouldn't take forever

![alt text][image10]

The validation accuracy goes up to 94.79% but the prediction time goes up to 17ms per image, with 729 windows that will take 12s per image (not including the feature extraction), so the full video could take about 4 hours to process, that's was out of the table... 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?
The code where generate this samples can be found in [task window search](src/task_window_search.py).

I used the same algorythm for optimized window search that we learnt in the classroom, you can see it here [hog window search.py](src/tools/hog_window_search.p)

The main optimization is performing the HOG only once per window size.

As you can see in the `combined_window_search` method from that file, I decided to do 5 passes with different window scales: x1, x1.5, x2, x2.5, and x4 in order to find cars of all sizes in the image. Smaller scales are only applied to higher zones in the image, because that's where we expect the cars to look smaller.

x1 (ymax-min: 380 -> 480)
![alt text][image11]

x1.5 (ymax-min: 380 -> 560)
![alt text][image12]

x2 (ymax-min: 380 -> 620)
![alt text][image13]

x2.5 (ymax-min: 380 -> 660)
![alt text][image14]

x4 (ymax-min: 380 -> 700)
![alt text][image15]

That gives a total of 729 windows, and seems to work pretty well, here's an example of how the windows detect cars in the image (sadly, with one false positive):

![alt text][image16]

#### 2. Heatmaps & threshold
The code where generating this sample images can be found in [task window search heatmap](src/task_window_search_heatmap.py).

I used the combination of the detected windows to generate a heatmap, then added a threshold of 4 (only pixels where 4 or more windows intersect will be detected as cars).

This is what we get after this processing.
![alt text][image17]

#### 3. Final pipeline (window search + heatmap + labels)
The code where generating this sample images can be found in [task window search pipeline](src/task_window_search_pipeline.py).

I used the `label` method from the `scipy.ndimage` library to extract the attention blobs from the heatmap, and create bounding boxes for the detected cars.

This is the final result of the pipeline, and what our video will look like:
![alt text][image18]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
The full 50s video took around 30 mins to be generated, at about 1.5 images per second (too slow for real time driving).

Here's a [link to my video result](./output_video.mp4)

I also tried the **rbf** model in a 5s sample of the video where the linear model has a hard time to detect the white car in the distance.
The **rbf** model performs very good, find a very fitter bounding box around both cars and has 0 false positives (with a threshold of 0, so very few false positives are generated and get discarded in the video processing flow).

The big issue with this is that it took 30 mins to process the 5 second video!!! Trying to process the full 50s video would take 5 hours! So of course I didn't do it 😅
Here's a [link to my video 5s result using rbf](./output_video_rbf.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to improve the result and remove as many false positives as possible, I created a [video processor](src/tools/video_processor.py) class that implements the full processing pipeline, and 10 heatmap images.
I then only take the pixels from those 10 heatmap images where the last 6 consequent frames detect a car (so for example if a pixel detects a car for 6 frames it will be kept, but if it doesn't detect a car for a single frame after that it will be discarded and only be used again if if detects a car again for 6 frames in a row).

I found that having a smaller window allowed the detection of cars where they are further away, but also generates a lot more of false positives.
Making the window bigger reduced the number of false positives but cars far away were not detected, and also it introduced a lot of lag to the image (24fps so a 6 images window adds 0.25s lag)

I didn't manage to get rid of all the false positives on the left-side of the image, but the result is not bad. The false posirtives only remain in the image for a few milliseconds. As I trick, knowing that in the video we're diving in the left lane, could be to discard all the pixels with x < 500, but that would only work for this example.

Here's a gif of the final video result:

![alt text][gif1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As I already mentioned, I didn't manage to get a perfect car detection (without false positives) using the LinearSVC, and using rbf took too long to generate predictions, so unless there is a way to parallelize those tasks (generate predictions for all windows at the same time) and use a very powerfull machine, it doesn't seem like this approach could be used in real time.

My following intuition would be to use a simple deep learning model, I belive it could perform MUCH better that the linear SVC with much faster predictions in order to approach the real-time (at least 10 images/second) 

The hardest part of this assignment was to find the proper features, not to much but not to few, to get a good enough accuracy without it being too slow. When using deep-learing most of this burden is taken care by the top convolutional layers.
The issue with DNN is that there's too many layers of obfuscation, so you end up not really knowing WHY it works. A much deeper study of what features to use with traditional ML algorythms could give better results that could be used in real-time car detection, and also keep a good understanding of what the machine is doing.
