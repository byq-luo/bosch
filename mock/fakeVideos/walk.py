import os

#here is some crap code that finds basic stats on the labels
#remember that kevin did mention that we can just forget about the barrier label if we want.

labels = {}
labelss = []
numfiles = 0
for dc in os.walk('./'):
    for dcc in dc:
        for fn in dcc:
            if fn in ['.','/'] or '.py' in fn:
                continue
            with open(fn) as f:
                numfiles += 1
                for line in f:
                    s = line.split(',')[0].split('=')[0]
                    labels[s] = labels.get(s,0) + 1
                    labelss.append(s)

print('There are',len(labelss),' labels spread across',numfiles,'files')
print('Label       | Frequency')
for label,freq in labels.items():
    print(label,' '*(10-len(label)),'|',freq)


# cutin
# evtEnd
# lcRel
# barrier
# cutout
# objTurnOff
# rightTO
# end
