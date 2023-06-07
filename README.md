```
mkdir pointpillars_ros
cd pointpillars_ros
git clone https://github.com/cat722/pp_realtime.git
catkin_make
```


#### 进入到搭建好的openpcdet环境
```python
conda activate pcdet
pip install pyquaternion
 
sudo apt-get install ros-melodic-jsk-recognition-msg
sudo apt-get install ros-melodic-jsk-rviz-plugins
```

在ros.py文件中修改预训练权重和config文件，改成你自己的路径即可。如果是你自己的雷达或相机219行换成你自己的话题名

#### 运行
```python
conda activate pcdet
source devel/setup.bash
roslaunch pointpillars_ros pointpillars.launch
```
