from math import sqrt

def calculate_accuracy(predictedLabels, correctLabels):
    # accuracyCouter = 0
    # if predictedLabel == correctLabel:
    #     accuracyCouter += 1

    correctPredictions = {k: predictedLabels[k] for k in predictedLabels if
                              k in correctLabels and predictedLabels[k] == correctLabels[k]}
    print("Total Number of Correct Predictions: ", len(correctPredictions))

    correctPredictionCount = len(correctPredictions)

    accuracy = (correctPredictionCount/ (len(predictedLabels))) * 100
    return accuracy * 0.01


#save pedictions and groundtruth as dictionaries
#key = time , value = label

predictions = {55:"cutIn", 76: "cutOut", 90: "end", 108: "end"}
groundTruth = {55:"cutIn", 76: "cutIn", 90: "end", 108: "end", 200: "end"}




# accuracy = total correct predictions / total predictions made * 100
accuracy = calculate_accuracy(predictions, groundTruth)
print("This is the accuracy:", accuracy)
z = 1.64  #number of standard deviations from the gaussian distribution (90%)
n = 500 #sample size
interval = z * sqrt( (accuracy * (1 - accuracy)) / n)  # binomial confidence interval

print("Interval: ", interval)