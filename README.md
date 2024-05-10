# AI 학습 모델 만들기.
roboflow 홈페이지에서 마음에 드는 모델을 다운 받아서 ai 모델을 만들어 봅시다.<br>
사이트 주소: <https://universe.roboflow.com/>

## 작업 환경(장치 사향)
장치 이름:	DESKTOP-O998J3H<br>
프로세서:	Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz   3.19 GHz<br>
설치된 RAM:	16.0GB<br>
장치 ID:	79976040-AE86-48D2-BE1C-72385C61EAC7<br>
제품 ID:	00330-80000-00000-AA923<br>
시스템 종류:	64비트 운영 체제, x64 기반 프로세서<br>
펜 및 터치:	이 디스플레이에 사용할 수 있는 펜 또는 터치식 입력이 없습니다.<br>

## Windows 사양
에디션:	Windows 10 Pro
버전:	22H2
설치 날짜:	‎2023-‎04-‎12
OS 빌드:	19045.4291
경험:	Windows Feature Experience Pack 1000.19056.1000.0

## 모댈
Yolov5

## 사용한 프로그램
Visual Studio Code, Python 3.9

## 제작 방법
  1. 폴더를 만듭니다. 예) C:\Users\WSU\Desktop\hardcaps
  2. 폴더에 들어가 우클릭 하고 VSCODE를 실행합니다.
  3. 터미널을 열어주세요. (ctrl + shift + ~)
  4. 아레의 명령어를 입력해서 가상환경을 만듭니다.
```

conda create -n yolov5 python=3.9

```
  6. python: 인터프리터 선택(ctrl + shift + p)을 통해 방금 만든 서버와 연결합니다.
  7. 연결 했다면 새 터미널을 엽니다.<br>
  -> (yolov5) C:\Users\WSU\Desktop\hardcaps (이런식으로 떳다면 성공)
  10. git clone으로 yolov5를 내려 받습니다.
```

git clone https://github.com/ultralytics/yolov5

``` 
  9. 'yolov5' Folder로 이동합니다. 
  10. 개발 환경에 필요한 Package를 설치합니다.
```

pip install -r requirements.txt

```

***

### 학습데이터 준비하기
1. myGlob.py 파일을 제작하고 아레에 있는 코드를 입력해 주세요.

```
 #여기서 주의 할 점은 데이터셋의 위치(경로)를 잘 맞추어 주세요 
 #실행하는 폴더에 따라서 상대 경로가 달라 지므로 절대 경로를 쓰는 것이 나을 수도 있습니다.
 #vaild 폴더가 없는 모델도 존재 할 수 있습니다. 그럴경우 test 폴더의 이름ㅇf vaild로 바꿔서 진행하시면 됩니다.

from glob import glob

train_img_list = glob('C:/Users/WSU/Desktop/hardcaps/yolov5/hardcaps/train/images/*.jpg')
test_img_list = glob('C:/Users/WSU/Desktop/hardcaps/yolov5/hardcaps/test/images/*.jpg')
valid_img_list = glob('C:/Users/WSU/Desktop/hardcaps/yolov5/hardcaps/valid/images/*.jpg')

print(len(train_img_list), len(test_img_list), len(valid_img_list))

if len(train_img_list) > 0:    
    with open('C:/Users/WSU/Desktop/hardcaps/yolov5/hardcaps/train.txt', 'w') as f:
        f.write('\n'.join(train_img_list) + '\n')
    with open('C:/Users/WSU/Desktop/hardcaps/yolov5/hardcaps/test.txt', 'w') as f:
        f.write('\n'.join(test_img_list) + '\n')
    with open('C:/Users/WSU/Desktop/hardcaps/yolov5/hardcaps/val.txt', 'w') as f:
        f.write('\n'.join(valid_img_list) + '\n')
```

2. data.yaml을 수정합니다. : <- 표시가 있는 것들의 경로만 수정합니다 나머지는 건들 필요 없습니다.
```
train: C:/Users/WSU/Desktop/hardcaps/yolov5/hardcaps/train/images
val: C:/Users/WSU/Desktop/hardcaps/yolov5/hardcaps/valid/images

nc: 3
names: ['head', 'helmet', 'person']
```

3. ./models/yolov5s.yaml을 복사해서 custom_yolov5s.yaml을 만들고 nc : 80을 data.yaml에 있는 nc 값으로 변경해 줍니다.
* 예시
```
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

# Parameters
nc: 3 # number of classes
depth_multiple: 0.33 # model depth multiple
width_multiple: 0.50 # layer channel multiple
anchors:
  - [10, 13, 16, 30, 33, 23] # P3/8
  - [30, 61, 62, 45, 59, 119] # P4/16
  - [116, 90, 156, 198, 373, 326] # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
    [-1, 3, C3, [128]],
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
    [-1, 6, C3, [256]],
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
    [-1, 9, C3, [512]],
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
    [-1, 3, C3, [1024]],
    [-1, 1, SPPF, [1024, 5]], # 9
  ]

# YOLOv5 v6.0 head
head: [
    [-1, 1, Conv, [512, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 6], 1, Concat, [1]], # cat backbone P4
    [-1, 3, C3, [512, False]], # 13

    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 4], 1, Concat, [1]], # cat backbone P3
    [-1, 3, C3, [256, False]], # 17 (P3/8-small)

    [-1, 1, Conv, [256, 3, 2]],
    [[-1, 14], 1, Concat, [1]], # cat head P4
    [-1, 3, C3, [512, False]], # 20 (P4/16-medium)

    [-1, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]], # cat head P5
    [-1, 3, C3, [1024, False]], # 23 (P5/32-large)

    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
  ]
```

***

### 학습
#### 모든 준비가 끝났다면 이제 터미널창에 명령어를 입력해 작업을 해봅시다.

1. 아레에 있는 코드의 괄호가 쳐져 있는 부분이 각각 data.yaml 파일과 custom_yolov5s.yaml 파일이 위치하고 있는 경로를 입력해 주세요.

```
python train.py --img 416 --batch 16 --epochs 100 --data (data.yaml의 경로) --cfg (custom_yolov5s.yaml의 경로) --weights '' --name _result --cache
```

2. 경로를 입력했다면 변경한 코드를 vscode의 터미널에 입력해 주세요.
3. 입력이 완료되었다면 이제 작업이 완료되길 기다려주면 됩니다.

<p>$\it{\large{\color{#DD6565} 주의사항!}}$</p>
<p>${\large{\color{#DD6565} 이작업은\ 상상\ 이상으로\ 컴퓨터의\ 메모리를\ 사용합니다.}}$</p>
<p>${\large{\color{#DD6565} 만약\ 작업이\ 도중에\ 멈추거나\ 하는\ 경우가\ 있다면\ 그건\ 메모리가\ 부족하다는\ 뜻입니다.}}$</p>
<p>${\large{\color{#DD6565} 그러니\ 해당\ 작업을\ 진행\ 중\일 때엔\ 웬만하면\ 다른\ 모든\ 창과\ 기능은\ 잠시\ 멈춰주세요.}}$</p>

### 학습이 끝난다면?

1. runs\train\pothole_result\weight에 있는 best.py 를 원하는 이름으로 변경해서 저장하면 됩니다.
2. 나중에 인터페이스나 프로젝트에 사용하시면 됩니다.
3. 끝╰(*°▽°*)╯

***

## 이곳에 있는 모델들의 데이터 파일 위치
[PotholeModel](https://public.roboflow.com/object-detection/pothole)<br>
[MoterbikeModel](https://universe.roboflow.com/pkc-4cbyd/-bmz60)<br>
[FiresmokeModel](https://public.roboflow.com/object-detection/wildfire-smoke/)<br>
[HardcapsModel](https://public.roboflow.com/object-detection/hard-hat-workers)<br>
[PKlotModel](https://public.roboflow.com/object-detection/pklot)<br>

하이퍼 링크를 클릭해 주세요. :-)
