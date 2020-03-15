from math import sqrt


def calculate_accuracy(predictedLabel, correctLabel, predictedLabelCount):
    accuracyCouter = 0
    if predictedLabel == correctLabel:
        accuracyCouter += 1
    accuracy = (accuracyCouter / predictedLabelCount) * 100
    return accuracy


#save pedictions and groundtruth as dictionaries
#key = time , value = label

predictions = {55:"cutIn", 76: "cutOut", 90: "end", 108: "end"}
groundTruth = {55:"cutIn", 76: "cutIn", 90: "end", 108: "end", 200: "end"}

correctPredictionCount = {k: predictions[k] for k in predictions if k in groundTruth and predictions[k] == groundTruth[k]}
print("Total Number of Correct Predictions: ", len(correctPredictionCount))


# accuracy = total correct predictions / total predictions made * 100
accuracy = .10
z = 1.64  #number of standard deviations from the gaussian distribution (90%)
n = 50 #sample size
interval = z * sqrt( (accuracy * (1 - accuracy)) / n)  # binomial confidence interval

print("Interval: ", interval)