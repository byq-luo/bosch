import os
import pickle
for _, _, filenames in os.walk('featuresLSTM'):
    files = filenames
print(len(files))
for i,filepath in enumerate(filenames):
    filepath= 'featuresLSTM/' + filepath
    print(i/len(files))
    with open(filepath, 'rb') as file:
        (rawboxes, boxscores, lines, lanescores, vehicles, boxcornerprobs) = pickle.load(file)
        with open(filepath, 'wb') as file:
            pickle.dump((boxscores, lanescores, vehicles, boxcornerprobs), file)
