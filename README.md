# 火焰识别
面对复杂环境中的火情侦测，人力物力成本十分高昂，为什么不用ai来解决这个问题呢

## 一、项目背景
希望通过开发一个基于深度学习能力的火险巡检系统，给传统的人工火险巡检带来一次变革，以更快的速度、更广的覆盖范围进行火灾潜在危险的预防，从而有效保障大众的生命安全。

## 二、数据集简介
采用ai studio自带数据集，约有2000份数据，均为火焰数据样本
```
import os
import random
 #用于分配训练集的比例权重，生成main下的各个文件
trainval_percent = 0.5
train_percent = 0.5
xmlfilepath = '/home/aistudio/dataset/Annotations'
txtsavepath = '/home/aistudio/dataset/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)
 
num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)
 
ftrainval = open(txtsavepath+'/trainval.txt', 'w')
ftest = open(txtsavepath+'/test.txt', 'w')
ftrain = open(txtsavepath+'/train.txt', 'w')
fval = open(txtsavepath+'/val.txt', 'w')
 
for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest .close()
```

## 三、模型选择和开发

### 3.1模型组网及可视化
```
import paddle.vision.transforms as T
network = paddle.nn.Sequential(
    paddle.nn.Flatten(),           # 拉平，将 (28, 28) => (784)
    paddle.nn.Linear(784, 512),    # 隐层：线性变换层
    paddle.nn.ReLU(),              # 激活函数
    paddle.nn.Linear(512, 10)      # 输出层
)
# 模型封装
model = paddle.Model(network)

# 模型可视化
model.summary((1, 28, 28))
```
### 3.2模型训练
```
!python tools/train.py -c configs/yolov3/yolov3_mobilenet_v3_large_ssld_270e_voc.yml --eval --use_vdl=True --vdl_log_dir="./output"
```
### 3.3预训练模型预测
```
!python work/PaddleDetection/tools/infer.py -c work/PaddleDetection/configs/yolov3/yolov3_mobilenet_v3_large_ssld_270e_voc.yml -o weights=work/PaddleDetection/output/yolov3_mobilenet_v3_large_ssld_270e_voc/model_final.pdparams --infer_img=dataset/JPEGImages/113.jpg
```
## 四、将预训练模型导出

```
!python work/PaddleDetection/tools/export_model.py -c work/PaddleDetection/configs/yolov3/yolov3_mobilenet_v3_large_ssld_270e_voc.yml \
        --output_dir=./inference_model \
        -o weights=work/PaddleDetection/output/yolov3_mobilenet_v3_large_ssld_270e_voc/model_final.pdparams
```

## 五、效果展示
![](https://ai-studio-static-online.cdn.bcebos.com/cb1276417603487f8cae20ac33f61f22e2b0db92a43e4c3ca3a075599ef58ca6)

<br>原图</br>

![](https://ai-studio-static-online.cdn.bcebos.com/f833091b2a224754815c371628f9cb6fb0f89ca50f5740b1bd36ff0bd0a87b2d)

<br>预测图</br>

## 六、心得体会
第一次写这些文案，写的也不是很好，由于自己也是第一次搞ai，避免不了比较大面积的借鉴，但是我认为自己还是对深度学习有了更深刻的了解吧：）
这次文案说实话挺一言难尽的，虽然不能说学了多少，学的有多会，但最起码还是会了一点应用（基于csdn训练），比小白强里一点吧

## 项目链接 Link of project：
https://aistudio.baidu.com/aistudio/projectdetail/2239594
## 个人简介 Learn about me:
我在AI Studio上获得青铜等级，点亮1个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/877171
