1 Introduction
Dataset. We will use Taskar’s OCR dataset for this project(the same dataset which was used in
Assignment2). The dataset description is repeated here again for your convenience.
The original dataset was maintained by Ben Taskar. It contains the image and label of 6,877 words
collected from 150 human subjects, with 52,152 letters in total. To simplify feature engineering,
1
CS 512 Lab 2: Conditional Random Fields with Convolutions
Figure 1. Example word image Figure 2. CRF for word-letter
each letter image is encoded by a 128 (=16*8) dimensional vector, whose entries are either 0 (black)
or 1 (white). Note in this dataset, only lowercase letters are involved, i.e. 26 possible labels. Since
the first letter of each word was capitalized and the rest were in lowercase, the dataset has removed
all first letters.
Minibatches. When training a model in PyTorch, it is common to send minibatches of training
examples to the optimizer. A minibatch is a small subset of the training data (the size of the
minibatch is usually a tunable hyperparameter).
In the starter code that is provided, the DataLoader is already setup to process the input dataset
as batches. The entire dataset is also divided evenly into training and test data in train.py.
However, when implementing the CRF and Convolution layer, you will have to be mindful of the
fact that the input to your module will be a minibatch. The following is the shape of the data after
batching.
X ∈ R
batch size×max word length×128 (1)
So every row of a batch corresponds to a word (sequence). Here max word length is the maximum
word length of all the words in the dataset. For words whose length is less than max word length,
zero padding is added. Note that while calculating the loss, you are supposed to exclude the padded
instances from the loss computation.
The entire dataset is divided into minibatches with same dimension (add zeros for nonexistent
letter in a word). This trick makes data loading/exporting easier, such as in data_loader and
get_conv_feature. However, it is a little bit troublesome for crf because we have to clamp the
zero-padded words. This results in one line of code for computing the valid lengths of words before
feeding the whole batch into convolution layers.
Conditional Random Fields. The CRF model is the same as what was defined in Assignment
2. However, there is a difference in the input that is passed to the CRF model. To recall the details
of the OCR dataset - the training set consists of n words. The image of the t-th word can be
represented as Xt = (x
t
1
, . . . , x
t
m)
0
, where 0 means transpose, t is a superscript (not exponent), and
each row of Xt
(e.g. x
t
m) represents a letter. Here m is the number of letters in the word, and x
t
j
is
a 128 dimensional vector that represents its j-th letter image. To ease notation, we simply assume
all words have m letters. The sequence label of a word is encoded as y
t = (y
t
1
, . . . , yt
m), where
y
t
k ∈ Y := {1, 2, . . . , 26} represents the label of the k-th letter. So in Figure 1, y
t
1 = 2, y
t
2 = 18, . . . ,
y
t
5 = 5.
2
CS 512 Lab 2: Conditional Random Fields with Convolutions
In this assignment, the CRF model will instead take convolutional features, given by a function g,
which you will implement. The details of the convolution operation is given in Section 3.
Using this (new) notation, the Conditional Random Field (CRF) model for this task is a sequence
shown in Figure 2, and the probabilistic model for a word/label pair (X, y) can be written as
p(y|X) = 1
ZX
exp Xm
s=1
hwys
, g(xs)i +
mX−1
s=1
Tys,ys+1!
(2)
where ZX =
X
yˆ∈Ym
exp Xm
s=1
hwyˆs
, g(xs)i +
mX−1
s=1
Tyˆs,yˆs+1!
. (3)
h·, ·i denotes inner product between vectors. Two groups of parameters are used here:
 Node weight: Letter-wise discriminant weight vector wk ∈ R
128 for each possible letter
label k ∈ Y;
 Edge weight: Transition weight matrix T which is sized 26-by-26. Tij is the weight associated with the letter pair of the i-th and j-th letter in the alphabet. For example T1,9 is the
weight for pair (‘a’, ‘i’), and T24,2 is for the pair (‘x’, ‘b’). In general T is not symmetric, i.e.
Tij 6= Tji, or written as T
0 6= T where T
0
is the transpose of T.
Given these parameters (e.g. by learning from data), the model (2) can be used to predict the
sequence label (i.e. word) for a new word image X∗
:= (x
∗
1
, . . . , x
∗
m)
0 via the so-called maximum
a-posteriori (MAP) inference:
y
∗ = argmax
y∈Ym
p(y|X∗
) = argmax
y∈Ym



Xm
j=1


wyj
, g(x
∗
)j

+
mX−1
j=1
Tyj ,yj+1



. (4)
When CRF is used as a layer, we need to compute the gradient with respect to its input. In (2),
let us denote zs = g(xs). Then
∇zs p(y|X) = wys −
X
yˆ∈Ym
exp(...)
Z
· ∇zs
Xm
s=1
hwyˆs
, zsi (5)
= wys −
X
yˆ∈Ym
p(yˆ|X)wyˆs
(6)
= wys −
X
yˆ∈Y
p(ys = ˆy|X)wyˆ. (7)
2 PyTorch
You will use the popular PyTorch deep learning framework to implement all the algorithms in
this assignment. For a comprehensive introduction to PyTorch please refer to this link PyTorch
Tutorial.
3
CS 512 Lab 2: Conditional Random Fields with Convolutions
The above tutorial is a beginner-friendly introduction to the basic concepts of PyTorch. You will
need to be comfortable with all the concepts in that link to successfully complete this assignment.
In particular, pay attention to the nn.module class. This is a standard way computational units
are programmed in PyTorch. You will implement the CRF layer and the Conv layer, in the code, as
a subclass of the nn.module class. Refer to the starter code for more details.
Like in the last assignment, Torch has a gradient checker. You could use it like this:
https://discuss.pytorch.org/t/how-to-check-the-gradients-of-custom-implemented-loss-function/8546
3 (20 points) Convolution
Different from the previous assignment, we are going to feed in convolutional features of the input
image of a letter to the CRF model. Your task is to implement the convolution layer in
PyTorch. Note that PyTorch implements its own convolution layer (nn.conv2d). You are required
to provide your own implementation and NOT use PyTorch’s implementation. However, you
may use PyTorch’s implementation of convolution as a reference to check the correctness of your
implementation.
Convolution operation. Convolution is a commonly used image processing technique, applying various types of transformations on an image. Convolutional Neural Networks (CNNs) employ
multiple layers of convolutions to capture fine-grained image features, which are further used downstream in learning several computer vision tasks such as object detection, segmentation etc.
A convolution operation takes in an image matrix X and a filter matrix K and computes the
following function as detailed in Eq 9.6 of [GBC]:
Xˆ(i, j) = X
k,l
X(i + k, j + l)K(k, l). (8)
In our case, the output channel from CNN is 1. Kernel is square: 5 × 5 or 3 × 3.
(3a) (20 points) Implement the Conv layer and the get conv features(x) function, in the starter
code. Once convolution is implemented, the CRF’s forward pass and loss functions use the
convolution features as inputs (the code for this is set up already).
Your implementation also needs to accommodate different strides, along with an option of
zero padding or not.
Testing your implementation. Your implementation of the Conv layer will contain the
implementation of the convolution operation. It is crucial to get the implementation of the
convolution operation correct first. Consider this simple example of an input matrix X and
filter matrix K, with unit stride and zero padding. Report the result of convolving the X
4
CS 512 Lab 2: Conditional Random Fields with Convolutions
with K. You must write this as a test case for the grader to run.
X =






1 1 1 0 0
0 1 1 1 0
0 0 1 1 1
0 0 1 1 0
0 1 1 0 0






; K =


1 0 1
0 1 0
1 0 1


Implement your test case inside a file conv test.py. It should run as a standalone test (with
all the dependencies, imports in place), and print the result on the screen.
Note. In PyTorch input to conv2d are 4-D tensors - (batch size × channel size ×
height × weight) for both the input image X and the filter K. In our dataset, we use
a single channel input (channel size = 1).
You only need to implement the forward pass of the convolution layer. There is no need to
implement the derivatives. You can use PyTorch’s auto-differentiation feature to automatically get backpropagation.
Having the backward() function implemented is an indication to PyTorch that the backward
pass is indeed implemented. If a layer in the model is specified to have the backward function,
then PyTorch will just use it. Otherwise, if some layers do not have the backward function
explicitly implemented, then PyTorch will use autograd to compute the gradients.
4 (50 points) CRF
Now, you will (re)-implement the CRF model in PyTorch. Note that this version is designed to
use convolutional features and NOT the raw pixels to the CRF model (recall in Assignment 2 we
used raw images pixels as the input features x
t
j which is a 128 dimensional vector). Here, they will
be replaced by convolutional features. However, the CRF implementation should remain almost
the same; except for changes in the input and output shapes and the fact that it needs to be
implemented as a layer (nn.module) in PyTorch.
The CRF model is implemented as a class in the crf.py file in the provided starter code.
(4a) Implement the forward, backward pass and loss inside crf.py. This would amount to
1. re-implementing the inference procedure using dynamic programming (decoder)
2. dynamic programming algorithm for gradient computation including with respect to
CRF input, T, and wy,
3. loss - which is the negative log-likelihood of the CRF objective.
You can directly copy from the reference solutions for the last assignment or
use your own implementation. Once again, place holders for all these are provided in
the starter code (in the crf.py file). This question will be graded through the subsequent
questions.
5
CS 512 Lab 2: Conditional Random Fields with Convolutions
(4b) (20 points) Implement and display performance measures on the CRF model - we will
use the same performance measures as the previous assignment (1) letter-wise prediction
accuracy, (2) word-wise prediction accuracy. Using a batch size of 64 plot the letterwise and word-wise prediction accuracies on both training and test data over 100 iterations
(x axis). (Place holder provided in the startup code). Use a 5 × 5 filter matrix for this
experiment, and set stride and zero/no padding to optimize the test performance. Initialize
the filters randomly. If it has not converged (function value changes little), you may increase
the number of iterations.
Note. Your model should process the input data batch after batch, therefore the input
dimension to your model is 256*14*128 (batch_size * max_word_length * num_of_pixels).
The output dimension of the CNN layer (i.e., the input dimension of the CRF layer) should
be 256*14*64, where 64 is the length of the features of one letter.
Note. In the backward function of CRF, should we calculate and return the gradient of
loss with respected to the whole batch of input, which is of dimension 256*14*64? We are
calculating the gradient of loss with respect to weights. But the loss is calculated using the
features and not the image itself.
Note. You have to make sure that invalid letters do not add any value to your loss function.
Moreover, while you are calculating the gradient, you have zero-out the gradients of invalid
letters. You cannot just use the true labels to find valid letters because, in test time, the labels
are not provided to the model. The solution was to use the input image and find valid images
(any image with at least one non-zero pixel). You can use ’torch.any()’ and ’torch.where()’
for masking and filtering.
(4c) (20 points) It is common to use more than one convolution layer in modern deep learning
models. Convolution layers typically capture local features in an image. Stacked convolutions
(multiple layers of convolutions put one after the other) help capture higher level features in an
image that has shown to aid classification significantly. Repeat experiments in (4b) with the
following convolution layers. Set stride and zero/no padding to optimize the test performance.
1. A Layer with 5 × 5 filter matrix
2. A Layer with 3 × 3 filter matrix
Note. In crf.py, the get_conv_features() function is merely a ”placeholder” for getting
the convolution features for the CRF model. The output shape of the conv layer will vary
depending on filter size, padding & stride. You are supposed to handle that (by zero padding)
and make the input tensor to the CRF as a fixed shape, after you get convfeatures inside
get_conv_features().
Note. In train.py, conv_shapes = [[1,64,128]] is a model parameter that specified the
input/embedding shapes. Feel free to ignore this parameter and define your own way of
handling the shapes. Here 64 is the embedding dimension, but if you are having multiple
convolution layers (stacked convolution layers), then the shapes of the subsequent layers will
be different. Batch_size is typically not specified in conv_shapes since its not a property of
the layer; no matter what shapes you specify, the definition of batch_size will not impact
them, the shape of any data (input/output) will be something like batch_size × a × b × c.
Note. In train.py, what does embed_dim = 64 mean? Typically layers are specified via
the input and output dimensions that it produces. Here input_dim corresponds to the input
6
CS 512 Lab 2: Conditional Random Fields with Convolutions
dimension that the layer takes in and the embed_dim is the size of the embedding (output)
that the layer produces. These are merely placeholders, meant to help you consider the input
and output shapes while programming the CRF layer (and subsequently the conv layer). If
this is confusing for you, feel free to setup your own mechanism to correctly handle input
output shapes. The question asks you to perform convolution with different filter shapes. So
one way to handle output shapes correctly is to let your convolution layer automatically infer
the output shape, given the input shape and filter size; some pointers:
https://fomoro.com/projects/project/receptive-field-calculator
Note. Should we use Sequential to concatenate convolution layer and crf layer, or implement
a CRF layer that entangles with a couple of convolution layers? Use Sequential. Pass a conv
layer (it could be a stack of sequential conv layers) to CRF and use the get_conv_features()
method to get the features. This way, we are able to easily determine the valid letters from
the input image as well.
(4d) (10 points) Enable GPU in your implementation. Does it lead to significant speedup? You
can test on the network in 4c. Make sure your plot uses wallclock time as the horizontal axis.
5 (30 points) Comparison with Deep Learning
Compare your new CRF model, with convolution features, with a convolution based Deep Neural
Network (DNN) model of your choice, also known as Convolutional Neural Networks (CNN). You
are free to design your own DNN model or pick one of the popular model architectures that have
been published. Some popular choices would be,
1. VGG [4]
2. ResNet [1]
3. AlexNet [2]
4. GoogLeNet [5]
5. LeNet [3]
Since all these methods, except LeNet, require resizing the images from 16x8 to 224x224, you can
just consider LeNet. You can use code from online, and build a blank LeNet and train all weights
from scratch. It shouldn’t be hard, and it will not take long. The input to the CNN model will of
course be the original train and test dataset. None of these methods are composed with a CRF.
You will have to report the following in your report.
(5a) (10 points) If you designed your own DNN model, then report the implementation details
of it, along with the model architecture, loss functions etc. If you picked LeNet, explain each
of the layers inside it and its purpose for the task at hand. That is, what functions (layers)
were useful in the solution to the problem. In addition, look into the source code and sketch
its structure within 150 words.
7
CS 512 Lab 2: Conditional Random Fields with Convolutions
(5b) (5 points) Plot the letter-wise and word-wise prediction accuracies on both training and test
data (y axis) over 100 iterations (x axis) (You might have to implement these). Compare this
with your CNN+CRF results and report your analysis (which model fared better? and why?).
You may use the hyperparameter that yields the best performance for your CNN+CRF model.
If it has not converged in 100 iterations, you may increase the number of iterations.
Sometimes, your program might run out of memory. In this case, you will have to adjust the
batch size. This post might help:
https://stats.stackexchange.com/questions/284712/how-does-the-l-bfgs-work
(5c) (5 points) Change the optimizer from LBFGS to ADAM. Repeat the experiments in (5b)
and report the letter-wise and word-wise accuracies, with x-axis as the #iterations. Does
ADAM find a better solution eventually, and does it converge faster than LBFGS?
(5d) (10 points) Why did you choose this model (again it could be your own design or an offthe-shelf model)? More precisely you should explain every design decision (use of batchnorm
layer, dropout layer etc) and how it helped in the task at hand, in your report.
