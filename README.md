## 开发环境
* 内存：110G
* 显存：16G
* 磁盘：20G
* python 版本：3.6.6
* python package : 
  - scikit-learn            0.19.2
  - tensorboard             1.8.0
  - tensorflow              1.9.0
  - tensorflow-gpu          1.8.0
  - Keras                   2.2.1
  - Keras-Applications      1.0.4
  - numpy                   1.16.0
  - pandas                  0.25.1
 * CUDA Version : 9.0.176
 * CUDNN Version:
```
  -bash-4.2$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
  #define CUDNN_MAJOR 7
  #define CUDNN_MINOR 2
  #define CUDNN_PATCHLEVEL 1
  --
  #define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
```

## 模型
模型采用resnet和densenet融合的方式，具体结构请查看net.info

## 源码文件说明
- train.py, 模型训练源码文件。执行该程序，自动从data/路径下读取训练数据，训练模型，并将训练好的模型保存到user_data/路径下。
- predict.py, 使用模型进行预测的源码文件。执行该程序，自动从data/路径下读取testB测试数据，调用训练出来的模型预测，将结果文件result.txt保存到prediction_result/下。
- main.sh， 自动执行训练和预测程序的脚本。

## 特殊说明
- 保证没有使用标注和id信息。
- 由于开发环境内存充足，所以数据都是直接全部加载到内存中，根据平时观察，高峰时期内存使用超过了60G，所以复现时请使用110G内存的机器。
- 模型训练时间较长，最好成绩的模型使用16G显存，训练时间在5~6小时。
- 已经上传最好成绩的模型文件ecg.model，如果复现困难，可以考虑跳过训练，将ecg.model放到user_data/路径下，然后执行predict.py复现结果。
- 训练过程中存在使用随机算法的场景，比如每轮训练前，对全量数据进行shuffle操作，所以重新训练出来的模型较之前的模型可能有一定偏差。

## 其他
- 由于工作繁忙，该模型没有进行太多调优，最终复赛成绩[54/2353]，队名：aierwiki
