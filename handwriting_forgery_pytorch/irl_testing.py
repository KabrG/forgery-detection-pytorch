from training import Net, CustomDataset, DataLoader, load_model, test
import sys
# Data will have a name with an associated label
img_dir = [sys.path[0]+"/IMG_4860.JPG"]
print(img_dir)
placeholder = [2]
print("reach 0")
dataset = CustomDataset(img_dir, placeholder)
print("reach 1")
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
print("reach 2")
print(data_loader)
# x, y = data_loader
model = Net().to('cpu')
load_model()
y_pred = test(data_loader, model)
print(y_pred.item())
# data_loader = DataLoader(dataset, batch_size=20, shuffle=True)  # returns two arrays: array of 100x100 and labels

