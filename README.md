# Convolution_Neural_Network

Convolutional Neural Networks (CNNs) are Artificial Intelligence
algorithms based on multi-layer neural networks that learns relevant
features from images, being capable of performing several tasks like
object classification, detection, and segmentation.

https://poloclub.github.io/cnn-explainer/



## Introduction
Image recognition is the task of taking an image and labelling it. For us humans, this is one of the first skills we learn from the moment we are born and is one that comes naturally and effortlessly. By the time we reach adulthood we are able to immediately recognize patterns and put labels onto objects we see. These skills to quickly identify images, generalized from prior knowledge, are ones that we do not share with our machines.

<p align="center"><img src="https://naushadsblog.files.wordpress.com/2014/01/pixel.gif", width="360"></p>
<p align="center">Fig 0.0 how a machine 'views' an image</p>

When a computer sees an image, it will see an array of pixel values, each between a range of 0 to 255. These values while meaningless to us, are the only input available to a machine. 

## Convolution neural networks
Image recognition used to be done using much simpler methods such as linear regression and comparison of similarities. The results were obviously not very good, even the simple task of recognizing hand-written alphabets proved difficult. Convolution neural networks (CNNs) are supposed to be a step up from what we traditionally do by offering a computationally cheap method of loosely simulating the neural activities of a human brain when it perceives images.

### Images
How computer sees a rgb image.

<p align="center"><img src="/imgs/rgb.jpg", width="100"></p>

How computer sees a gray image.

<p align="center"><img src="/imgs/gray.jpg", width="100"></p>

### CNNs explained
But first, let us understand what a convolution is without relating it to any of the brain stuff.

#### The mathematical part

<p align="center"><img src="/imgs/input-image-dimension.JPG", width="240"></p>
<p align="center">Fig 1.0 simplified depiction of a 32x32x3 image</p>

A typical input image will be broken down into its individual pixel components. In the picture above, we have a 32x32 pixel image which has a R, G, and B value attached to each pixel, therefore a 32x32x3 input, also known as an input with 32 height, 32 width, and 3 depth.

<p align="center"><img src="/imgs/filtering.JPG", width="360"></p>
<p align="center">Fig 1.1 applying a 3x3 filter</p>
<p align="center"><img src="/imgs/filtering-math.JPG", width="720"></p>
<p align="center">Fig 1.2 mathematics of filtering</p>

A CNN would then take a small 3x3 pixel chunk from the original image and transform it into a single figure in a process called filtering. This is achieved by multiplying a number to each of the pixel value of the original image and summing it up. A simplified example of how the math is done is as described in the picture above. NOW STOP RIGHT HERE! Make sure you understand the mathematics of how to conduct filtering. Re-read the contents if you need to. As for how we arrive at this filter and why it is of the size 3x3, we will explain later in this article.

Since we are dealing with an image of depth 3 (number of colors), we need to imagine a 3x3x3 sized mini image being multiplied and summed up with another 3x3x3 filter. Then by adding another constant term, we will receive a single number result from this transformation.

<p align="center"><img src="/imgs/filtering-many-to-one.gif", width="360"></p>
<p align="center">Fig 1.3 filtering in action, original image is below</p>

This same filter will then be applied to every single possible 3x3 pixel on the original image. Notice that there are only 30x30 unique 3x3 squares on a 32x32 image, also remember that a filter will convert a 3x3 pixel image into a single image so the end result of applying a filter onto a 32x32x3 image will result in a 30x30x1 2nd 'image'.

#### The high-level explanation

What we are trying to do here is to detect the presence of simple patterns such as horizontal lines and color contrasts from the original image. The process as described above will output a single number. Typically this number will be either positive or negative. We can understand positive as the presence of a certain feature and negative as the absence of the feature.

<p align="center"><img src="/imgs/finding-horizontal-vertical.jpg", width="540"></p>
<p align="center">Fig 1.4 identifying vertical and horizontal lines in a picture of a face</p>

In the image above, a filter is applied to find vertical and horizontal lines and as we can see, in each of the pictures on the left, only the places where vertical lines are present will show up in white and likewise horizontal lines for the picture on the right.

Going by this idea we can think of filtering as a process of breaking down the original image into a list of presence of simplified structures. By knowing the presence of slanted lines and horizontal lines and other simple basic information, more interesting features such as eyes and nose and mouth then then be identified. If the presence of eyes, mouth and nose are detected, then the classifier will have a pretty high certainty that the image at hand is probably a face. Basically that is what a CNN would do, by doing detective work on the abstract information that it is able to extract from the input image and through a somewhat logical thought process come to the deduction of the correct label to attach to a particular image. The model might not exactly look for eyes or nose, but it would attempt to do something similar in an abstract manner.

Make sure that you have understood all that was covered previously because the next section is going to progress at a much faster rate. We are still not going to talk about how to calculate filters yet but first, let us finish up the mechanics of the CNN.

#### Back to the model

One filter would only be capable of finding a single simplified feature on the original image. Multiple filters can be applied to identify multiple features. Let's say on the original image, a total of 32 filters are applied on the input 32x32x3 image. One filter applied onto the image will result in a 30x30x1 output. So to apply 32 unique filters, you merely stack the outputs on top of one another to result in a 30x30x32 output.

The entire process of transforming an input from a 32x32x3 form to a 30x30x32 form is known as a single convolution layer. An entire CNN model is usually made up of multiple convolution layers and a classifier layer. Here is an example of how a typical CNN would look like.

<p align="center"><img src="/imgs/conv-layers.jpeg", width="720"></p>
<p align="center">Fig 1.5 structure of a typical CNN, here classifying a car</p>

The model would take an input from the left (here the image of a car). The data will be transferred from the left side to the right, through each of the individual layers. Each layer would take the output of the previous layer as its input and then produce a transformation on the image before passing it onto the next layer. There are probably a few terms that you might not understand at this point of time, but let us go through them one at a time:

- CONV: In the model in the picture, the first layer is a CONV layer. It is nothing new as CONV is just short form for convolution layer.

> There are of course convolution layers of different sizes and not just 3x3. Some models uses 7x7 and even 11x11 filters but larger filters also mean more parameters which means longer training time. Filters are also usually has odd lengths and are squares. This is so as to have some sort of center to take reference from. There is also another concept called strides. In the examples above we use strides of size 1. Stride 2 would mean starting from the top left most 3x3 section of the image, you move 2 pixels to the right before you apply your filter again, the same when you move downwards. Padding is another technique commonly used. Usually when filtering takes place, the original image would shrink. If you pad the original image with pixels of values of 0 around it's borders, you will effectively be able to maintain image size. ie 32x32 input 32x32 output (instead of 30x30). This is crucial if you wish to build deep learning models which contains more layers.

- RELU: The RELU layer (short for rectifier layer) is basically a transformation of all negative outputs of the previous layer into 0. As negative numbers would also contribute to the output of the next layer, 0 has a significance in the sense that it will not affect the results of the next layer. Looking back at the high-level definition of how a convolution works, negative numbers should mean the absence of a feature. 0 would fit that idea more concisely and that is the purpose of this layer. We will not change the values of the positive numbers as the magnitude of the positive number can help identify how closely the image represents a feature. The RELU layer will not transform the shape of it's input. If the input is of shape 30x30x32, the output would still be 30x30x32, except all the negatives are now 0s instead.

> In actual fact rectifiers are just a member of a larger family called activators, they all set out to achieve the same purpose as stated above. Another popular activation layer is the logistic activator, It transform it's inputs into a logistic distribution.

<p align="center"><img src="/imgs/max-pooling.jpeg", width="540"></p>
<p align="center">Fig 1.6 pooling on a 4x4 input</p>

- POOL: Image processing is a very computationally intensive process. To allow our algorithm to run at a decent speed while not compromising accuracy too heavily, we do a form of reduction on the image size in a technique called pooling. The image above shows how it is done. From each 2x2 square, we find the pixel with the largest value, retain it and throw away all the unused pixels we also do this for each depth layer (recall on the input image, it would be each color layer). Doing this transformation would essentially reduce the dimensions of the original image by half on height and another half on weight. Another reason we wish to do this is to converge features of close proximity together such that more complex features can develop sooner.

> The pooling technique we describe here is called max-pooling because we are only taking the max of every 2x2 squares. There are also other pooling methods such as min pooling and mean pooling. But this is by far the most popular method of pooling. Pooling can also be for larger dimensions like 3x3 or 4x4 although it is not recommended as image size will reduce too fast.

The act of repeating the process of CONV RELU POOL would simulate the process of identifying more complex features from the original image.

- FC: After retrieving all of the advanced features from each image, we combine them together to classify the image to it's proper label. We do so in the fully connected layer.

<p align="center"><img src="/imgs/fully-connected-layer.JPG", width="540"></p>
<p align="center">Fig 1.7 A simple fully connected layer displaying probability outputs</p>

The fully connected layer will take in all of the advanced features produced by the final convolution layer and output the probability for each label. Remember that the purpose of the convolution layers are to output the presence of advanced features such as eyes, mouth, or wings. By taking note of the presence of such features, the fully connected layer will do the last bit of detective work to determine the most suitable label to apply to each image. Mathematically, it works in the same way as filters do except this time, there are no 3x3 portions. Each 'filter' in this case will be the same size as the output layer from the final layer of convolution. There can, however, be multiple fully-connected-layers but only just as many as the number of label classes you have, the intuition being that you can calculate the confidence level of each individual class separately.

Do keep in mind, this is just a very basic understanding of what the fully connected layer seeks to accomplish. In actuality this layer can be much more complex but first, a long awaited question should first be answered.

### Where filter weights come from

#### The objective statement

> Short recap: Up to this current moment in time, your understanding of how CNNs work is that through a series of multiplications, summations and modifications, and you are able to generate a prediction of some sort. Along the way, complex features that a computer would not normally be able to identify are extracted and turned into simple terms that it could, these terms represent whether a high level feature is present or not. This greatly simplifies the original problem of image identification into small simple steps that a computer can solve but there's just one mystery that remains.

CNN is an algorithm that requires some very specific parameters (called weights) in the filter layers else the entire model would fail to function. We find these parameters using Mathematics.

The problem is this,

> _to find a set of parameters that allows the model to be as accurate at labelling images as possible_

To translate this into mathematics, let us first define a few terms,

<dl>
  <dt><img src="/imgs/x.JPG", width="30"></dt>
  <dd>Represents the original image</dd>

  <dt><img src="/imgs/y.JPG", width="30"></dt>
  <dd>Represents the actual label of the image</dd>

  <dt><img src="/imgs/y-hat.JPG", width="30"></dt>
  <dd>Represents the predicted label of the image</dd>

  <dt><img src="/imgs/y-hat.JPG", width="30"></dt>
  <dd>Represents the \`series of multiplications, summations and modifications\` the CNN makes on the the image to output you predicted value</dd>

  <dt><img src="/imgs/y-hat2.JPG", width="80"></dt>
  <dd></dd>
</dl>

Taking note of these definitions, we can also define our predicted y as follows,

> <p><img src="/imgs/y-hat2.JPG", width="80"></p>

When you take the predicted result and subtract it from our actual result, you get this back,

> <p><img src="/imgs/residual.JPG", width="80"></p>

One way of interpreting this is by viewing it as a measure of how far off the model is from the desired result (this measure is hereby called error). An error of 0 would mean that the model is spot on, 1 and -1 would mean that there are still improvements to be made. By averaging up the errors a CNN's predictions make on a set of images, you will be able to get a gauge of how well a set of parameters are doing. The greater the average error, the more inaccurate the predictions are, which prompts you to change the current set of parameters.

<p>Lets take the example of the case where we have 3 images. Suppose the errors of an algorithm trying to predict the actual labels of these images are 0, 1, and -1. If we sum up all these errors we should get the total error so 0 + 1 + (-1) = ... 0? Even if we average it out it would still be 0.
<!-- <img src="/imgs/you-dont-say.jpg", width="40"></p> -->

That does not mean that the predictions the CNN made are all correct. The problem lies in the method error is accumulated. As there are both positive and negative errors, they will cancel each other out but thankfully simple modification will fix this. By squaring the errors you will force all errors to be positive.

> <p><img src="/imgs/summation-symbol.JPG", width="30">, this symbol just means summation. In the context below, it means for all images, sum up (the term inside)</p>

From this,

> <p><img src="/imgs/sum-residual-wrong.JPG", width="80"></p>

we will get this,

> <p><img src="/imgs/sum-residual-square.JPG", width="90"></p>

and of course to account for averaging,

> <p><img src="/imgs/average-squared-error.JPG", width="100"></p>

So errors of 0, 1, and -1 will sum up to be (0^2) + (1^2) + ((-1)^2) = 0 + 1 + 1 = 2. Averaging that out will give us 2/3. The smaller this figure is, the closer we are to the optimal set of parameters. Therefore minimizing this term would be the same as finding the optimal parameters for the CNN. Minimization also has a symbol for convenience.

> <p><img src="/imgs/cost-function.JPG", width="120"></p>

Then lets replace our predicted y.

> <p><img src="/imgs/cost-function2.JPG", width="120"></p>

Take note that here, x and y are both fixed based on the input images you have provided the CNN with. There only thing we can change to minimize this equation is A, the parameters of all the layers of filters in the CNN. If minimizing this equation also means making the CNN more accurate, then that would be the same as solving the original English problem.

> _to find a set of parameters that allows the model to be as accurate at labelling images as possible_

The question of how we arrive at the optimal filter is still unanswered but to solve this,

> <p><img src="/imgs/cost-function2.JPG", width="120"> (also known as the cost function)</p>

there is an area of Mathematics dedicated to solving problems such as this called gradient descent.

#### Gradient descent

Let us first plot a simple graph.

<p><img src="/imgs/graph1.jpg", width="320"></p>

Imagine this, you have a horizontal axis. This axis represents every single unique possible combination of parameters for the CNN, _A_, all mapped out onto a line. Each point on this axis represents a unique _A_.

The vertical axis represents the average error at that specific _A_ (the cost in terms of model inaccuracy therefore the name cost function). As one can expect, the average error will not be a linear function, rather it will be curved, like in the image above.

<p><img src="/imgs/graph2.jpg", width="320"></p>

Recall that minimizing this average error will result in a more accurate model. Therefore, the point where the curve dips lowest corresponds to the set of parameters which allows the model to perform best. The problem of finding this point can be solved using gradient descent. Sadly there is no simple way to explain how the process of gradient descent work without watering it down too much. In summary it goes a little something like this,

1. Start off at a randomly initialized A
2. Calculate the average error generated by some neighboring A and move to a neighbor with the lowest A.
3. Repeat this process multiple times until you reach the A with the lowest average error.

Those of you familiar with calculus should be able to recognize that solving this problem involves finding differential functions. But those who aren't don't have to worry too much as most deep learning libraries these days are capable of doing these math for you. Learning the math is tedious especially for people without prior mathematical knowledge however it is still useful and fundamental when building more complex algorithms and models. By the way, the full cost function (average error) would also contain a regularization term as well as some other sophistications depending on the problem at hand.

One thing of note is that we do not specify the objectives for each filter. That is because the filters usually adjust themselves to identify complex features. This isn't exactly surprising from a statistical standpoint. Eyes, nose, and mouth are usually very good indicators in face identification. Models are considered good if they are able to identify abstractions of such complex features.



## Visualizing your CNN
An important skill to have is to be able to interpret models. Here we will cover 4 of such methods. Do note that I have used a deeper model (which requires longer training time) in the codes below as they generally give better visualization. You can load the model I used from ```./models/stashed/``` but it would be completely fine to use the model trained from the previous section.

These last few sections are left intentionally short. They are going to be relatively unguided, only a basic intuition of what needs to be done is given. The reason behind this is so that you can get right down to coding and researching the ways of implementation. If that isn't your cup of tea, then you can always just read through this and look at some of the pretty images I've plot out and run the codes I've done, I'll include the codes and how to run them below.

### Result based

On your console,
```
python visualize_results.py
```

It is not difficult to imagine how to visualize results based on how well a model performs but here are a list of things you can do,

- calculate model accuracy
- plotting out random images from the test set and printing the prediction made by the model
- plotting out wrongly predicted images
- plotting out a breakdown of wrongly predicted images

### Pixel-importance based

On your console,
```
python visualize_pixel_importance.py
```

You can also visualize which regions the model believes are important in making an accurate prediction. One way to do this is described in the steps below,

1. start with a correctly predicted image (it is important that it is correctly predicted since we know that the algorithm is probably capable of capturing it's key features)
2. remove a pixel or a section from the original image (I did by sections in ```visualize_pixel_importance.py```)
3. make predictions on the new image and see how much the removed aera contributed to making the correct prediction
4. plot out a heat map of how much each area contributes to making the prediction correct

<p align="center"><img src="/imgs/importance_1.jpeg", width="240"></p>
<p align="center">Fig 3.0 image of a dog, important areas shaded in red</p>

Depending on which pictures you used and the color scheme you used, you might end up with something like this. As you can see, important regions usually centered around the dogs ears, eyes and mouth. I apologies for the picture quality being like this the red parts are simply not coming out well. If anyone has any suggestion on making heat maps, please send me an email which can be found below!

### Activation based

On your console,
```
python visualize_hidden_layer_activation.py
```

After training your model, you can also attempt to visualize exactly what each filter is attempting to do. One method is through the construction of an input image which would maximize the output of a filter. In essence what this would achieve is the recreation of the feature that the filter gets most excited over (what the filter is attempting to find). Exactly how this is done is through gradient ascent (opposite of descent). Thankfully Keras can take care of the mathematics for us. A guide on how to do this along with some sample codes are available on [Keras's official blog](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html).

> Here you can also challenge yourself to learn gradient ascent and write your own algorithm to create these images

Here I have plotted out some images which would maximize the activation for 4 filters in each odd numbered convolution layer (this is done so as to save space and maintain objectivity).

<p align="center"><img src="/imgs/activation_1.jpeg", width="560"></p>
<p align="center">Fig 3.1 activation of convolution layer 1</p>

Layer 1:
- As we might expect, filters in layer 1 are looking for simple features. In this case, they are looking for unique colors.

<p align="center"><img src="/imgs/activation_3.jpeg", width="560"></p>
<p align="center">Fig 3.2 activation of convolution layer 3, more complex features are developing such as lines at different orientations</p>

Layer 3:
- Here is where things become more interesting. Filters above are attempting to detect lines of different tilt and colors.

<p align="center"><img src="/imgs/activation_5.jpeg", width="560"></p>
<p align="center">Fig 3.3 activation of convolution layer 5, filters can be seem attempting to find ball shapes</p>

Layer 5:
- A filter can clearly be seen built for the purpose of finding red balls, however from this point on features are starting to become too abstract to fully understand.

<p align="center"><img src="/imgs/activation_7.jpeg", width="560"></p>
<p align="center">Fig 3.4 activation of fully connected layer 1</p>

Layer 7:
- It is unclear what exactly these filters are attempting to look for as the level of abstraction is too high.


### Partial-output based

On your console,
```
python visualize_hidden_layer_output.py
```

Another way to visualize what filters are attempting to do is by plotting out the partial output after each convolution layer. The intuition is that partial outputs are the indicators for the presence of certain features (recall [the high-level explanation](#The-high-level-explanation)). After identifying a suitable image, all you have to do is to run the image through the layers one at a time and plot out those partial outputs.

Here we have an image of a truck, lets take a look at what each filter is attempting to detect.

<p align="center"><img src="/imgs/output_1.jpeg", width="180"></p>
<p align="center">Fig 3.5 original image of a truck</p>

<p align="center"><img src="/imgs/output_2.jpeg", width="560"></p>
<p align="center">Fig 3.6 partial output of convolution layer 2, high-levels of activation are colored in yellow</p>

Layer 1:
- We know from the previous visualization that this layer is attempting to locate colors. The filters that attempt to detect white are getting excited over the body of the truck while those which attempt to locate orange are excited over the head light.

<p align="center"><img src="/imgs/output_4.jpeg", width="560"></p>
<p align="center">Fig 3.7 partial output of convolution layer 3</p>

Layer 3:
- Here filters are getting excited over more complex features. Some filters appear to be detecting wheels and others seem to be attempting to find doors and windows.

Another thing to note is that partial outputs in convolution layer 3 is significantly smaller that those from convolution layer 1. This is due to the effects of pooling. Thus this method of visualization is suitable only for earlier layers as the deeper you go, the lower the resolution of the partial outputs.

