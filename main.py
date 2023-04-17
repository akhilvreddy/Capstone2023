import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchmetrics
import time
import ssl
from tqdm import tqdm
from sklearn.metrics import classification_report


ssl._create_default_https_context = ssl._create_unverified_context

img_W = 256
img_H = 256
input_shape = (3, img_W, img_H)
num_classes = 38

print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.Resize((img_W, img_H)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((img_W, img_H)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# using the ResNet NN
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


model = ResNet(num_classes)

train_p = r'insert directory'
test_p = r'insert directory'
valid_p = r'insert directory'

trainset = torchvision.datasets.ImageFolder(root=train_p, transform=transform_train)
testset = torchvision.datasets.ImageFolder(root=test_p, transform=transform_test)
validset = torchvision.datasets.ImageFolder(root=valid_p, transform=transform_test)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=33, shuffle=False)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
opt = optim.SGD(model.parameters(), lr=0.0002)

target_names = ['Apple scab', 'Apple Black rot', 'Apple Cedar rust', 'Apple healthy', 'Blueberry healthy', 'Cherry (including sour) Powdery mildew', 'Cherry (including sour) healthy', 'Corn (maize) Cercospora leaf spot Gray leaf spot', 'Corn (maize) Common rust', 'Corn (maize) Northern Leaf Blight', 'Corn (maize) healthy', 'Grape Black rot', 'Grape Esca (Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)', 'Grape healthy', 'Orange Haunglongbing (Citrus greening)', 'Peach Bacterial spot', 'Peach healthy', 'Bell pepper Bacterial spot', 'Bell pepper healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two-spotted spider mite', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    train_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass').to(device)
    start_time = time.time()
    with tqdm(total=len(train_loader)) as pbar1:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc.update(output.argmax(dim=1), target)
            pbar1.update(1)
    avg_loss = train_loss / len(train_loader.dataset)
    acc = train_acc.compute() * 100
    end_time = time.time()
    print('Train Epoch: {} Loss: {:.6f} Acc: {:.2f}% Total Execution Time: {:.2f}s'.format(epoch, avg_loss, acc,(end_time-start_time)))


def validate(model, device, valid_loader):
    model.eval()
    valid_loss = 0
    valid_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass').to(device)
    with tqdm(total=len(train_loader)) as pbar2:
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                valid_loss += loss.item()
                valid_acc.update(output.argmax(dim=1), target)
                pbar2.update(1)
    avg_loss = valid_loss / len(valid_loader.dataset)
    acc = valid_acc.compute() * 100
    print('Validation set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(avg_loss, acc))


def test(model, device, test_loader, target_names):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with tqdm(total=len(train_loader)) as pbar3:
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                y_true.extend(target.tolist())
                y_pred.extend(pred.view(-1).tolist())
                pbar3.update(1)
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    print(f'Classification report:\n{report}')
    for i, name in enumerate(target_names):
        print(f'{i}: {name}')
    return test_loss, accuracy, report

if __name__ == '__main__':
    for epoch in range(1, 11):
        train(model, device, train_loader, opt, epoch)
        validate(model, device, valid_loader)
        test(model, device, test_loader, target_names)
