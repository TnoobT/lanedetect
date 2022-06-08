# lanedetect-mnn
# ”Towards Light-weight and Real-time Line Segment Detection“ 的MNN实现

# 系统环境
1. 系统为macbookpro M1
2. MNN的版本为1.2.0

# 说明
1. https://github.com/lhwcv/mlsd_pytorch直接转出来的模型为 mlsd_no_max_sigmoid.mnn; 把sigmoid和maxpooling放入网络的forward中的模型为mlsd_with_max_sigmoid.mnn。
2. 网络原本输出[1,17,256,256],我只取了有用的五维[1,7:12,256,256]; 因此MNN模型的输出维度为[1,7:12,256,256]

# 提示
1. 请自己更改CMakeLists.txt中的路径，并替换掉lib中动态库。

# 运行
```shell
mkdir build & cd build
cmake ..
./lane
```
