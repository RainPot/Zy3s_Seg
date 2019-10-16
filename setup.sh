#!/usr/bin/env bash
echo "CS&model downloading..."
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/cityscapes_formated.tar
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/ResEMANet.pth
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/resnet101-5d3b4d8f.pth
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/highorder8c/Res59800.pth
echo -e "down!\nDecompression CS..."
tar -xf cityscapes_formated.tar
echo -e "down!\nstart training!"
python3 -m torch.distributed.launch --nproc_per_node=2 train2.py

