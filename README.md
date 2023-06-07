### This repository is based on OpenPCDet for ros adaptation, please install openpcdcet first

```
mkdir pointpillars_ros
cd pointpillars_ros
git clone https://github.com/cat722/pp_realtime.git
catkin_make
```

```python
conda activate pcdet
pip install pyquaternion
 
sudo apt-get install ros-melodic-jsk-recognition-msg
sudo apt-get install ros-melodic-jsk-rviz-plugins
```


#### 运行
```python
conda activate pcdet
source devel/setup.bash
roslaunch pointpillars_ros pointpillars.launch
```
Then just have a fun!

## TODO List
Real-time tracking based on detection results
