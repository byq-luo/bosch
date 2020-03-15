from math import sqrt


def calculate_accuracy(predictedLabel, correctLabel, predictedLabelCount):
    accuracyCouter = 0
    if predictedLabel == correctLabel:
        accuracyCouter += 1
    accuracy = (accuracyCouter / predictedLabelCount) * 100
    return accuracy


# accuracy = total correct predictions / total predictions made * 100
accuracy = .10
z = 1.64  #number of standard deviations from the gaussian distribution (90%)
n = 50 #sample size
interval = z * sqrt( (accuracy * (1 - accuracy)) / n)  # binomial confidence interval

print("Interval: ", interval)