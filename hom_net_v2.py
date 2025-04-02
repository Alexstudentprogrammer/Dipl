import np
import torch
import torch as t
import glob
import cv2
import torchvision.datasets as datasets
import torchvision.transforms as tr
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from coordconv import CoordConv2d


def perform_Transformation(image, M, ans):
    dst = cv2.warpPerspective(image, M, (225, 170))

    plt.figure(figsize=(20, 7))
    plt.subplot(121)
    plt.imshow(np.array(ans, dtype=int))
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(122)
    plt.imshow(np.array(dst, dtype=int))
    plt.axis('off')
    plt.title("Transformed Image")
    plt.show()

class CustomTensorDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __getitem__(self, index):
        x = torch.load(self.file_paths[index])
        return x

    def __len__(self):
        return len(self.file_paths)


def my_transform():
    return tr.Compose([
        tr.ToTensor(),
    ])


device = torch.device("cuda")

transform = tr.ToTensor()
ans_files = glob.glob("ans/tensor/*.pt")
train_files = glob.glob("train_hom/tensor/*.pt")

test_ans_files = glob.glob("test_ans_hom/tensor/*.pt")
test_ans_img_files = glob.glob("test_ans_hom/tensor_img/*.pt")
test_files = glob.glob("test_hom/tensor/*.pt")

print(len(ans_files))

# train_data = datasets.ImageFolder(root='train_hom', transform=my_transform())
train_data = CustomTensorDataset(file_paths=train_files)
ans_data = CustomTensorDataset(file_paths=ans_files)

print(train_data)

train_loader = DataLoader(train_data, batch_size=5, shuffle=False)
ans_loader = DataLoader(ans_data, batch_size=5, shuffle=False)

test_data = CustomTensorDataset(file_paths=test_files)
test_ans_data = CustomTensorDataset(file_paths=test_ans_files)
test_ans_img_data = CustomTensorDataset(file_paths=test_ans_img_files)

test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
test_ans_loader = DataLoader(test_ans_data, batch_size=1, shuffle=False)
test_ans_img_loader = DataLoader(test_ans_img_data, batch_size=1, shuffle=False)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.pool = nn.MaxPool2d(2)

        self.coord = CoordConv2d(6, 10, 1, use_cuda=True, with_r=True)

        self.encode1 = nn.Sequential(
            nn.Conv2d(10, 80, 3, padding=1),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(inplace=True),
        )
        self.encode1_2 = nn.Sequential(
            nn.Conv2d(80, 35, 3, padding=1),
            nn.BatchNorm2d(35),
            nn.LeakyReLU(inplace=True),
        )

        self.encode2 = nn.Sequential(
            nn.Conv2d(35, 2, 3, padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(76500, 150),
        )
        self.fc2 = nn.Linear(150, 8)

    def forward(self, X):
        X = self.coord(X)

        X = self.encode1(X)
        X = self.encode1_2(X)
        X = self.pool(X)
        X = self.encode2(X)

        X = torch.flatten(X, 1)
        X = self.fc1(X)
        X = nn.functional.leaky_relu(X)
        X = self.fc2(X)
        return X


model = NN().to(device)
model.load_state_dict(torch.load('hom_approximation_v2_test.pth'))
criterion = nn.L1Loss(reduction='sum')

# learning_rate = 0.0005
# epochs = 9
# for i in range(epochs):
#
#     if epochs == 1:
#         learning_rate = 0.00003
#     elif epochs == 4:
#         learning_rate = 0.000007
#     elif epochs == 6:
#         learning_rate = 0.000001
#
#     optimizer = t.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001)
#     total_loss = 0
#     counter = 0
#
#     for b, (dataloders, mask) in enumerate(zip(train_loader, ans_loader)):
#
#         b += 1
#         counter += 1
#         print(counter)
#         dataloders = dataloders.to(device)
#         mask = mask.to(device)
#
#         y_pred = model(dataloders).to(device)
#
#         loss = criterion(y_pred, mask)
#         total_loss += loss.item()
#
#         if counter % 100 == 0:
#             print("current loss: ", loss, b, i)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     torch.save(model.state_dict(), 'hom_approximation_v2_test.pth')
#     print(total_loss / 4000)

# point 1
x_1 = 145
y_1 = 242

# point 2
x_2 = 308
y_2 = 242

# point 3
x_3 = 308
y_3 = 121

# point 4
x_4 = 145
y_4 = 121

total_loss_test = 0
graph = []
for b, (data, mask, mask_img) in enumerate(zip(test_loader, test_ans_loader, test_ans_img_loader)):

    data = data.to(device)
    mask = mask.to(device)
    y_pred = model(data).to(device)
    y_pred = y_pred[0].detach().to("cpu")

    point_x = mask_img[0][0]
    point_y = mask_img[0][1]

    ans_x = mask_img[0][2]
    ans_y = mask_img[0][3]

    # restoring homography
    x_1_new = x_1 + y_pred[0]
    y_1_new = y_1 + y_pred[1]

    x_2_new = x_2 + y_pred[2]
    y_2_new = y_2 + y_pred[3]

    x_3_new = x_3 + y_pred[4]
    y_3_new = y_3 + y_pred[5]

    x_4_new = x_4 + y_pred[6]
    y_4_new = y_4 + y_pred[7]

    H, status = cv2.findHomography(np.array([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]]),
                                   np.array([[x_1_new, y_1_new], [x_2_new, y_2_new], [x_3_new, y_3_new],
                                             [x_4_new, y_4_new]]),
                                   cv2.RANSAC
                                   )
    point = np.array([point_x, point_y, 1])
    new_point_homogen = np.matmul(H, point)

    predicted_point_x = new_point_homogen[0] / new_point_homogen[2]
    predicted_point_y = new_point_homogen[1] / new_point_homogen[2]

    error = ((float(abs(predicted_point_x - ans_x))**2 + float(abs(predicted_point_y - ans_y)))**2)**0.5
    # print(((float(abs(predicted_point_x - ans_x))**2 + float(abs(predicted_point_y - ans_y)))**2)**0.5)
    data = data.cpu().detach().numpy()
    data = data[0]

    data = np.transpose(data, (1, 2, 0))
    if error < 1000:
        graph.append(error)
        total_loss_test += error
        
print("Mean error: ", total_loss_test / 999)
plt.plot(graph)
plt.show()
