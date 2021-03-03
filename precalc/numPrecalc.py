import os
import pickle
print("GRID POINTS:")
for root, dirs, files in os.walk("/home/rehmemk/git/anugasgpp/Okushiri/precalc/precalcValues"):
    for file in files:
        if file.endswith(".pkl"):
            filename = os.path.join(root, file)
            with open(filename, 'rb') as fp:
                data = pickle.load(fp, encoding='latin1')
            print(f"{file} contains {len(data.keys())} keys")

print("MC VALUES:")
for root, dirs, files in os.walk("/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues"):
    for file in files:
        if file.endswith(".pkl"):
            filename = os.path.join(root, file)
            with open(filename, 'rb') as fp:
                data = pickle.load(fp, encoding='latin1')
            print(f"{file} contains {len(data.keys())} keys")
