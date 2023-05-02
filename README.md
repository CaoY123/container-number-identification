## 注意事项
1. 使用gpu训练的命令
```
python train.py --weights yolov7x.pt --cfg cfg/training/yolov7-up.yaml --data data/up.yaml --batch-size 3 --epoch 300 --device 0
```
注意上面的 --batch-size 不能设的大了，否则会报内存空间分配不足的 error，经过实践，为1时可以正常运行

注：  
1). 可以通过命令：tensorboard --logdir=runs 来查看训练过程中的日志  
2). 在已经有权重文件的情况下，可以不运行上述命令进行训练 

2. 标注图片命令
```
python detect.py --weights best.pt --source xxx --device 0
```

3. 裁剪已经标定的图片的命令，--source后传入要裁剪的已经有标定框的图片，--label后传入的标定框的坐标文件
```
python cut.py --source XXXX --label XXXXX
```

4. 对于裁剪的图片预处理命令
```angular2html
python pretreat.py --source XXXX
```

5. 对于预处理后的图片进行分割
```angular2html
python devide.py --source XXXX
```

6. 生成训练模板匹配识别模型图片的脚本 generate.py，其根据./fonts文件夹下的字体  
   生成不同的数字和大写英文字母的白色字体黑色背景的大小为512 * 512 的二值化图片，  
   以用于训练模板匹配识别经过处理的包含单个编号字符的二值化图片

