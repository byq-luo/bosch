import os

bufferTO = 10
buffer = 5
groundTruthPath = "D:\entire_hdd\labels_correct_timestamp"
genereatedPath = r"C:\Users\jgeng\Desktop\labels_final2"
#genereatedPath = r"D:\entire_hdd\labels"
#genereatedPath = r"D:\labels"

groundPaths = []
generatedPaths = []
totalGroundLables = 0
correctLabels = 0
extraLabels = 0
successfulIterations = 0
difference = 0
ground_list ={}
extra_list = {}
missed_list = {}
correct_list = {}
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
        #totalGroundLables += len(groundLabels)
    except:
        continue

    for label1, labelTime1 in groundLabels:
        found = False
        if label1 in ground_list:
            ground_list[label1] += 1
        else:
            ground_list[label1] = 1
        totalGroundLables += 1
        for label2, labelTime2 in genereatedLabels:
            if label1 == "rightTO":
                if (label1, labelTime1) in groundLabels:
                    if label1 == label2 and labelTime1 - bufferTO < labelTime2 < labelTime1 + bufferTO:
                        found = True
                        difference += abs(labelTime2 - labelTime1)
                        correctLabels += 1
                        if label1 in correct_list:
                            correct_list[label1]+=1
                        else:
                            correct_list[label1] = 1
                        genereatedLabels.remove((label2, labelTime2))
                        if (label1, labelTime1) in groundLabels:
                            groundLabels.remove((label1, labelTime1))
                        continue
            else:
                if (label1, labelTime1) in groundLabels:
                    if label1 == label2 and labelTime1 - buffer < labelTime2 < labelTime1 + buffer:
                        found = True
                        difference += abs(labelTime2 - labelTime1)
                        correctLabels += 1
                        if label1 in correct_list:
                            correct_list[label1]+=1
                        else:
                            correct_list[label1] = 1
                        genereatedLabels.remove((label2, labelTime2))
                        if (label1, labelTime1) in groundLabels:
                            groundLabels.remove((label1, labelTime1))
                        continue

        if found is False:
            if label1 in missed_list:
                missed_list[label1] += 1
            else:
                missed_list[label1] = 1
    for label2, labelTime2 in genereatedLabels:
        extraLabels += 1
        if label2 in extra_list:
            extra_list[label2] += 1
        else:
            extra_list[label2] = 1



    successfulIterations += 1

adjustedGroundTotal = totalGroundLables
if "barrier" in missed_list:
    adjustedGroundTotal -= missed_list["barrier"]
#if "evtEnd" in missed_list:
    #adjustedGroundTotal -= missed_list["evtEnd"]
#adjustedGroundTotal -= missed_list["rightTO"]
print("total ground truth labels: ", totalGroundLables, "\ntotal corect genereated labels: ", correctLabels, " \ntotal extra labels: ", extraLabels, "\nvideos tested: ", successfulIterations, "\npercent right: ", correctLabels/adjustedGroundTotal, "\navg extra labels per video: ", extraLabels/successfulIterations, "\navg ground truth labels per video: ", totalGroundLables/successfulIterations, "\navg time difference from ground truth label: ", difference/correctLabels)
print("extra labels: ", extra_list)
print("missed labels: ", missed_list)
print("ground truth label count: ", ground_list)
print("correct labels: ", correct_list)


