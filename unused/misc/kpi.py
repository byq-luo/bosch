import os

bufferTO = 10
buffer = 5
groundTruthPath = "D:\entire_hdd\labels_correct_timestamp"
genereatedPath = r"C:\Users\jgeng\Desktop\labels"
groundPaths = []
generatedPaths = []
totalGroundLables = 0
correctLabels = 0
extraLabels = 0
successfulIterations = 0
difference = 0
# groundPaths = Storage.recursivelyFindVideosInFolder(self, groundTruthPath)

for folderPath, subdirs, files in os.walk(groundTruthPath):
    for file in files:
        name, ext = os.path.splitext(file)
        groundPaths += [folderPath + '\\' + file]

for folderPath, subdirs, files in os.walk(genereatedPath):
    for file in files:
        name, ext = os.path.splitext(file)
        generatedPaths += [folderPath + '\\' + file]

for item in groundPaths:
    groundLabels = []
    genereatedLabels = []
    lines = []
    try:
        with open(item, 'r') as f:
            lines = f.readlines()
        for ln in lines:
            label, labelTime = ln.split(',')
            label = label.split('=')[0]  # handle rightTO label
            labelTime = float(labelTime)
            groundLabels.append((label, labelTime))
    except:
        continue
    labelPath = item.replace(groundTruthPath, genereatedPath)
    lines = []
    try:
        with open(labelPath, 'r') as f:
            lines = f.readlines()
        for ln in lines:
            label, labelTime = ln.split(',')
            label = label.split('=')[0]  # handle rightTO label
            labelTime = float(labelTime) % 300
            genereatedLabels.append((label, labelTime))
    except:
        continue

    for label1, labelTime1 in groundLabels:
        for label2, labelTime2 in genereatedLabels:
            if label1 == "rightTO":
                if label1 == label2 and labelTime1 - bufferTO < labelTime2 < labelTime1 + bufferTO:
                    difference += abs(labelTime2 - labelTime1)
                    correctLabels += 1
                    genereatedLabels.remove((label2, labelTime2))
                    continue
            else:
                if label1 == label2 and labelTime1 - buffer < labelTime2 < labelTime1 + buffer:
                    difference += abs(labelTime2 - labelTime1)
                    correctLabels += 1
                    genereatedLabels.remove((label2, labelTime2))
                    continue

    for label2, labelTime2 in genereatedLabels:
        extraLabels += 1

    successfulIterations += 1
    totalGroundLables += len(groundLabels)

print("total ground truth labels: ", totalGroundLables, " corect genereated labels: ", correctLabels, " extra labels: ", extraLabels, " iterations: ", successfulIterations, " percent right: ", correctLabels/totalGroundLables, "avg extra: ", extraLabels/successfulIterations, " avg difference: ", difference/correctLabels)



