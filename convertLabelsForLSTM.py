import os


"""
    Timestamp for formatted data is rounded to the nearest tenth of a second then multiplied by 10.

"""
def _loadFromDisk():

    #for filename in os.listdir(r"C:\Users\jgeng\Desktop\bosch_stuff\labels"):
        #labelsPath = r"C:\Users\jgeng\Desktop\bosch_stuff\labels\\" + filename
    for filename in os.listdir('precomputed/groundTruthLabels'):
        groundTruthLabels = []
        formattedGroundTruth = []
        labelsPath = "precomputed/groundTruthLabels/" + filename
        print(labelsPath)
        try:
            with open(labelsPath) as file:
                labelLines = [ln.rstrip('\n') for ln in file.readlines()]
                for ln in labelLines:
                    label, labelTime = ln.split(',')
                    label = label.split('=')[0]
                    labelTime = (float(labelTime))
                    temp = labelTime/300
                    correctTime = (temp - int(temp))*300
                    groundTruthLabels.append((label, correctTime))
        except:
            groundTruthLabels = []

        #print(groundTruthLabels)

        for item in groundTruthLabels:
            timestamp = int(round(item[1], 1) * 10)
            timestamp = timestamp / 10
            if item[0] == "rightTO":
                tup = (0, timestamp)
                formattedGroundTruth.append(tup)
            elif item[0] == "lcRel":
                tup = (1, timestamp)
                formattedGroundTruth.append(tup)
            elif item[0] == "cutin":
                tup = (2, timestamp)
                formattedGroundTruth.append(tup)
            elif item[0] == "cutout":
                tup = (3, timestamp)
                formattedGroundTruth.append(tup)
            elif item[0] == "evtEnd":
                tup = (4, timestamp)
                formattedGroundTruth.append(tup)
            elif item[0] == "objTurnOff":
                tup = (5, timestamp)
                formattedGroundTruth.append(tup)
            elif item[0] == "end":
                tup = (6, timestamp)
                formattedGroundTruth.append(tup)
            elif item[0] == "barrier":
                tup = (7, timestamp)
                formattedGroundTruth.append(tup)
            else:
                print ('we missed a label')
                print(item[0])

        print(formattedGroundTruth)
        newpath = "precomputed/convertedGroundTruthLabels/" + filename
        f = open(newpath, "w+")
        for item in formattedGroundTruth:
            f.write(str(item[0])+","+str(item[1])+"\n")
        f.close()

if __name__ == '__main__':
    _loadFromDisk()