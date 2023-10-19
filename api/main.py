from fastapi import FastAPI, File, UploadFile 
import uvicorn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import io
from PIL import Image 


# CNNの定義
#ver2.0のネットワーク
class CNN(nn.Module):
  def __init__(self, n_output, n_hidden):
    super(CNN,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=3,padding=1)
    self.maxpool1 = nn.MaxPool2d((2,2))
    self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=3,padding=1)
    self.maxpool2 = nn.MaxPool2d((2,2))
    self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
    self.conv4 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1)
    self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3)
    self.flatten = nn.Flatten()
    self.l1 = nn.Linear(9216,4096)
    self.relu1 = nn.ReLU(inplace=True)
    self.l2 = nn.Linear(4096,n_hidden)
    self.relu2 = nn.ReLU(inplace=True)
    self.l3 = nn.Linear(n_hidden,n_output)

    self.features = nn.Sequential(
        self.conv1,
        self.maxpool1,
        self.conv2,
        self.maxpool2,
        self.conv3,
        self.conv4,
        self.conv5

    )

    self.classifire = nn.Sequential(
        self.l1,
        self.relu1,
        self.l2,
        self.relu2
    )

  def forward(self, x):
    x1 = self.features(x)
    x2 = self.flatten(x1)
    x3 = self.classifire(x2)

    return x3

# 画像の前処理クラス
class DataTransform():
    def __init__(self):
        self.transform = transforms.Compose(
            [ transforms.Resize(32),
              transforms.ToTensor(),
              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ]
        )

    def __call__(self,data):
        return self.transform(data)


# 正解ラベル
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

n_output = len(classes)  
n_hidden = 128  

#DataTransformのインスタンス化
transform = DataTransform()
#ネットワークのインスタンス化         
net = CNN(n_output=n_output, n_hidden=n_hidden)
#学習済みパラメータのロード cpuの記述必要
net.load_state_dict(torch.load('model_ver2.0.pth', map_location=torch.device('cpu')))


app = FastAPI()

# uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=60)

# 通信の確認
@app.get('/')
async def conect_check():
    return {'message':'success'}

#アップロードされた画像ファイル名を返すエンドポイントの作成
@app.post('/uploadfile')
async def uploadfile(upload_file: UploadFile = (...)):
    return {
        'filename': upload_file.filename
    }

# POSTリクエストに対して推論結果を返す
@app.post('/predict')
async def predict(upload_file: UploadFile = (...)):
    #画像読み込みの処理いまいち理解できてない
    contents = await upload_file.read()
    input = Image.open(io.BytesIO(contents))
    input = transform(input)
    input = input.unsqueeze(0)

    #推論モード
    net.eval()

    with torch.no_grad():
        output = net(input)
    
    predicted_class = torch.argmax(output, dim=1)

    return {
        'result_class': classes[predicted_class]
    }


