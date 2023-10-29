import csv
import os

# Directories
real_dir = 'handwriting_dataset/authentic_signatures'
fake_dir ='handwriting_dataset/fake_signatures'
# Dataset of Real Handwriting
real_files = os.listdir(real_dir)
real_labels = [1]*len(real_files)
# Dataset of Fake Handwriting
fake_files = os.listdir(fake_dir)
fake_labels = [0]*len(fake_files)

all_files = real_files + fake_files
all_labels = real_labels + fake_labels

print(all_files)
print(all_labels)

with open('handwriting_dataset.csv', 'a', newline='') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(("Pixel Array", "Label"))
            for i in range(len(all_files)):
                csvwriter.writerow((all_files[i], all_labels[i]))