CS512 - Lab2:

Team Members:
Amrit Raj Vardhan,
Harsh Mishra,
Karan Jogi,
Manmohan Dogra,
Nikita Thakur

Below is the structure of the code as per the assignment questions:

#Q3a:
`code/conv_test.py` contains the code to initialise the test case given in the assignment. We have used our implementation of 2D convolution in `conv.py` to print the result as standard output. 
We also cross-check our results with PyTorch's implementation of convolution function in `torch.nn.functional.conv2d`.

#Q4

##(b) 
The implementation for this question is present in `code/train.py`. 
It uses `crf.py` to initialise the model and print the training accuracies as the performance metric. It can be run using the below command:

```
$ python code/train.py
```

##(c)

The main difference is adding a new convolution layer of kernel size 3 to the model in 4b. for this, we need to uncomment the additional convolution layer code in crf.py and then
run the below command:

```
$ python code/train.py
```

We have included the comparison of the performance of this model with the one in 4(b) as a graph in the report.

##(d)
This question requires a performance comparison of GPU against CPU. We have done this by changing the device in 4b and 4c and using the time values we get from them to plot the comparison. 
The results are plotted using `code/plot_4d.py`. 
Results can be found in `code/cpu_time.txt`, `code/gpu_time.txt` 
The comparison graph can be found in the report.


#Q5

##(a) & (b)
The implementation for this question is present in `code/lenet_lbfgs.py`. 
It uses modified `data_loader.py` which can be found in `code/data_loader_q5.py`.
`code/plot.py` is used to plot the accuracies of test and train set. 
Results can be found in `code/lbfgs_files` folder.

##(c)
The implementation for this question is present in `code/lenet_adam.py`. 
It uses modified `data_loader.py` which can be found in `code/data_loader_q5.py`.
`code/plot.py` is used to plot the accuracies of test and train set. 
Results can be found in `code/Adam_files` folder.



