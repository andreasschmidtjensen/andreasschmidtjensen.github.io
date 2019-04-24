---
layout: post
title: Rubber Duck
blog_title: "Run or Walk?"
tldr: "Can I use my phone to tell me whether I'm running or walking (if, for some reason, I don't know)?"
---

I found a [blog-series about detecting a person's activity](https://towardsdatascience.com/run-or-walk-detecting-user-activity-with-machine-learning-and-core-ml-part-1-9658c0dcdd90): are they running or walking? The author built an app and started collecting data when he was walking or running, and labelled it accordingly. I downloaded the data and build a simple model to see how easily it can be distinguished. In itself, this is not advanced at all: the data is quite simple and the model is simple as well. 

However, what I like most is not just the modeling, but building the entire application using a model. I therefore wanted to try using Tensorflow.js to use the model in a small web-app that fetches data from a phone's sensors. 

The notebook is available [here](https://colab.research.google.com/github/andreasschmidtjensen/andreasschmidtjensen.github.io/blob/master/examples/run-or-walk/Run_or_Walk.ipynb) and the small website can be viewed [here]({{ site.baseurl }}/examples/run-or-walk/).

![Run or walk?]({{ site.baseurl }}/img/run-or-walk.png)

Below are just some notes that helped me do this.

## Getting sensor data
Getting sensor data is actually quite easy: 
```
window.addEventListener("devicemotion", function(event) {
    console.log(
        event.accelerationIncludingGravity.x + ", " +
        event.accelerationIncludingGravity.y + ", " +
        event.accelerationIncludingGravity.z
    );
});
```

However, it turns out that in order to actually retrieve the data, the webpage must be using HTTPS (that's the case for iOS, at least). This led to some headache at first, because I couldn't quite understand why it wouldn't work locally. 

I installed a `local-ssl-proxy` to be able to test locally:
```
npm install -g local-ssl-proxy
local-ssl-proxy --source 4001 --target 4000
```

Now I can access my webpage from my phone and start predicting my activity.

## Using Tensorflow.js
Assume we have a Keras model that we save in HDF5 format.
```
model = tf.keras.models.Sequential([...])
...
model.save("runorwalk.h5")
```

Converting to Tensorflow.js is done using the `tensorflowjs` python package, so we install it and then run the command to convert:
```
pip install tensorflowjs
tensorflowjs_converter --input_format keras runorwalk.h5 .
```

The result is a `model.json` file which contains the architecture and configuration of the model and a number of *binary* files that contain the weights (`group<x>-shard<y>of<z>.bin`).

### Using the model in Javascript
The model is loaded as follows:
```
const model = await tf.loadLayersModel("/examples/run-or-walk/model.json");
```
By specifying the absolute path, Tensorflow is able to also find the weights-files and load them accordingly.

Prediction is then done by creating a *tensor* and calling the `predict` method:
```
const data = [[1,2,3,4,5,6]];
const tensor = tf.tensor(data);
const result = await model.predict(tf.tensor(data));
```

The result is a tensor itself and the data can be fetched using the `array()` or `arraySync()` methods:
```
const output = result.arraySync()[0]
```

In my case, I just have to check if the output predicts I am walking (is less than `0.5`) or running.