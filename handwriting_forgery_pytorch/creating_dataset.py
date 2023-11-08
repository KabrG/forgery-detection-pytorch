import csv
import os
import cv2
import sys

def create_dataset_string():
    # Directories
    real_dir = '/Volumes/Lexar/Datasets/Real Handwriting'
    fake_dir ='/Volumes/Lexar/Datasets/Fake Handwriting'
    # Dataset of Real Handwriting
    real_files = os.listdir(real_dir)
    print(real_files)
    # Remove Underscore Bug
    for i in range(len(real_files)):
        real_files[i] = os.path.join(real_dir, real_files[i])

    real_labels = [1]*len(real_files)

    # Dataset of Fake Handwriting
    fake_files = os.listdir(fake_dir)
    for i in range(len(fake_files)):
        fake_files[i] = fake_dir + '/' + fake_files[i]

    fake_labels = [0]*len(fake_files)


    all_files = real_files + fake_files
    all_labels = real_labels + fake_labels

    for i in range(len(all_files)):
        all_files[i] = all_files[i].replace('._', '')
    print("data = ", all_files)
    print("labels = ", all_labels)


    class EmptyFileException(Exception):
        pass

    for x in all_files:
        try:
            img = cv2.imread(x)
            print("opened ", x)
            if img is None:
                print("NONE")
                raise EmptyFileException("Empty File")

        except EmptyFileException:
            print("Force Quit")
            sys.exit(2)

        except cv2.error:
            print("Force Quit")
            sys.exit(3)

        except Exception:
            print("Force Quit")
            print("couldn't open ", x)
            sys.exit(1)
    print("Files verified.")
    return all_files, all_labels
# with open('handwriting_dataset.csv', 'a', newline='') as file:
#             csvwriter = csv.writer(file)
#             csvwriter.writerow(("Pixel Array", "Label"))
#             for i in range(len(all_files)):
#                 csvwriter.writerow((all_files[i], all_labels[i]))
# create_dataset_string()