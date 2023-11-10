from training import Net, CustomDataset, DataLoader, load_model, test
import sys
import torch

# Image directory from content root
img_dir = '/IMG_4928.JPG'
img_dir = [sys.path[0]+img_dir] # Use system path
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # lr is learning rate
load_model(model, optimizer)
y_pred = test(data_loader, model)

print('Predicted value:', y_pred.item())
# data_loader = DataLoader(dataset, batch_size=20, shuffle=True)  # returns two arrays: array of 100x100 and labels

