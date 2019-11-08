#!/usr/bin/env bash
echo "CS&model downloading..."
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/cityscapes_formated.tar
#hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/ADEChallengeData2016.zip
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/ResEMANet.pth
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/resnet101-5d3b4d8f.pth
echo -e "down!\nDecompression CS..."
tar -xf cityscapes_formated.tar
#unzip -q ADEChallengeData2016.zip
echo -e "down!\nstart training!"
mkdir results
#python3 -m torch.distributed.launch --nproc_per_node=4 train_ADE20K.py

