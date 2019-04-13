---
layout: post
title: Rubber Duck
blog_title: "Facial Keypoints Detection"
tldr: "I found a dataset for facial keypoints detection and built an app adds sunglasses to my face on the webcam."
---

I came across a blog post about detecting facial keypoints using Convolutional Neural Networks (CNNs) and thought it would be fun to try as well. The blog post can be found [here](https://medium.com/datadriveninvestor/facial-key-point-detection-88ccfaeaf9ee). (Spoiler: I do many of the same things done in that, so don't expect any extraordinary findings here - I just wanted to experiment).

First, I created the CNN that can detect facial keypoints. All of that is described in [this notebook](https://colab.research.google.com/github/andreasschmidtjensen/facial-keypoints/blob/master/Facial_Keypoints_Detection.ipynb). Then, I used the model to do the detection realtime on my webcam.

## Finding faces
Loading the existing model is straightforward:
```
with open("facial-keypoints-detection.json", "r") as file:
    model = tf.keras.models.model_from_json(file.read())
model.load_weights('facial-keypoints-detection.h5')
```

The first issue arise when fetching data from the webcam. The training is based on well-cropped images that only show the face, but that is not what we will see from the webcam. Luckily, OpenCV comes with a [Cascade classifier for detecting faces](https://docs.opencv.org/3.2.0/db/d28/tutorial_cascade_classifier.html). 

```
# read frame from webcam
frame = webcam.read()
image = imutils.resize(frame, height=480)

# the model was trained on grayscale, so convert
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```

Each detected face can then be cropped out of the image and is ready for detection:
```
for (x, y, w, h) in detected_faces:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (96, 96))
```

We then use the model to find the keypoints of each face:
```
predictions = model.predict(faces)
```

Finally, we use the keypoints for the eyes to figure out where to put the sunglasses:
```
right_eye_corner = points[10] * f_w + f_x, points[11] * f_h + f_y
left_eye_corner = points[6] * f_w + f_x, points[7] * f_h + f_y
image = add_sunglasses(image, right_eye_corner, left_eye_corner)
```

Here, `f_w`, `f_h`, `f_x` and `f_y` corresponds to the width, height, x and y coordinates for the detected face. We use this to transform the predicted locations into locations that make sense in the full image (from the webcam).

<video controls>
    <source src="{{ site.baseurl }}/img/sunglasses.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

The full source is available here: [https://github.com/andreasschmidtjensen/facial-keypoints](https://github.com/andreasschmidtjensen/facial-keypoints)