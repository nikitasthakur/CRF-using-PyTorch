import matplotlib.pyplot as plt

#Just change the folder name from lbfgs_files to Adam_files to plot results for Adam optimizer.

letterwise_training = []
letterwise_testing = []
wordwise_training = []
wordwise_testing =  []

with open('4c/letterwise_training.txt') as f:
    lines = f.readlines()

    for line in lines:
        #uncomment for question 5
        # letterwise_training.append(float(line.split("\n")[0].split("tensor(")[1].split(",")[0]))
        
        letterwise_training.append(float(line.split("\n")[0]))
 
with open('4c/letterwise_testing.txt') as f:
    lines = f.readlines()

    for line in lines:
        #uncomment for question 5
        # letterwise_testing.append(float(line.split("\n")[0].split("tensor(")[1].split(",")[0]))
        letterwise_testing.append(float(line.split("\n")[0]))
 
with open('4c/wordwise_training.txt') as f:
    lines = f.readlines()

    for line in lines:
        wordwise_training.append(float(line.split("\n")[0]))
 
with open('4c/wordwise_testing.txt') as f:
    lines = f.readlines()

    for line in lines:
        wordwise_testing.append(float(line.split("\n")[0]))
 


iterations = [x for x in range(len(wordwise_testing))]


plt.figure(figsize=(12,10))

plt.plot(iterations, letterwise_training, color='red', linewidth=3,  label="Letterwise Training",linestyle='-')
plt.plot(iterations, letterwise_testing, color='red', linewidth=3, label="Letterwise Testing",linestyle=':')
plt.plot(iterations, wordwise_training, color='purple', linewidth=3, label="Wordwise Training",linestyle='-',)
plt.plot(iterations, wordwise_testing, color='purple', linewidth=3,  label="Wordwise Testing",linestyle=':')
plt.legend(loc = "upper left")

plt.ylim(0, 1)

plt.xlabel("Iterations")
plt.ylabel("Accuracy")

plt.title("Accuracy vs Iterations for 4c with convolutions")

plt.savefig("accuracies_4c.png")
