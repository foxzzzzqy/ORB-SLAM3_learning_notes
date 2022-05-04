
这是从csdn复制过来的，当时实验室github老上不去不敢挂梯或者改host，所以用的csdn
图太多了，格式不对。 欢迎访问我的csdn
https://blog.csdn.net/qq_39533374/article/details/123709139?spm=1001.2014.3001.5501


写得很烂，渣学校研一，啥也不懂，就看了十四讲和概率机器人，ros会调个包。希望能坚持下来

///???麻了，草稿不保存啊，发布了慢慢更新吧。

//去看每一个库函数、看懂所有代码不是很现实。半个月只看了三个部分。除了主体几个函数其他的应该看流程和大致代码。需要改的时候看具体哪个参数和具体的库函数。但是例如opencv提取特征点之类的函数看完之后应该好好学一下

//???为什么发布不了没显示哇写了好久的没了

//回环部分看的很难受不是很懂。尤其坐标变换啥的基本看不动。可能还需要过一下十四讲。后续可能会过下因子图优化

//看完了 后面的回环草草过了，看了一个多月了真看吐了啊。先看论文吧，如果打算做回环部分的再回头看

//这篇笔记虽然写的很烂，但对我来说收获肯定很大。对别人或者后面的学弟学妹只能当一下参考文档或踩坑笔记。其实应该做一下思维导图和参数名注释为了后面查询和改代码的，但是我懒而且别人做的也很好。

//一个月辣，5w字，润了润了，奖励自己打一晚上游戏。



目录

安装教程：

 遇到和解决的问题

参考资料：

前期准备：

 笔记：

main函数

System.cc

ORBextrsvtor.cc

rbrief描述子旋转不变性

brief描述子：

特征匹配

void ExtractorNode::DivideNode

ORBextractor::DistributeOctTree

ORBextractor::ComputeKeyPointsOld/

MapPoint 

void MapPoint::Replace(MapPoint* pMP)替换地图点

void MapPoint::ComputeDistinctiveDescriptors()

void MapPoint::UpdateNormalAndDepth()

float MapPoint::GetMaxDistanceInvariance()

Frame.cc

Frame::Frame

Frame::Frame

void Frame::AssignFeaturesToGrid()

void Frame::ExtractORB

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)

vector Frame::GetFeaturesInArea

void Frame::ComputeStereoMatches()

Frame::Frame 

KeyFrame

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)

void KeyFrame::UpdateBestCovisibles()

void KeyFrame::UpdateConnections(bool upParent)

void KeyFrame::SetBadFlag()

tracking

void Tracking::MonocularInitialization()

void Tracking::CreateInitialMapMonocular()

void Tracking::StereoInitialization()

void Tracking::Track() 

1.恒速模型估计位姿

2.参考帧估计位姿

3.重定位估计位姿

bool Tracking::TrackLocalMap()

bool Tracking::NeedNewKeyFrame()

void Tracking::CreateNewKeyFrame()

LocalMapping

主函数void LocalMapping::Run()

void LocalMapping::ProcessNewKeyFrame()处理关键帧

void LocalMapping::MapPointCulling()剔除不好的地图点

void LocalMapping::CreateNewMapPoints()

void LocalMapping::SearchInNeighbors()融合当前关键帧和其共视帧的地图点

LoopClosing闭环

主函数run 

bool LoopClosing::NewDetectCommonRegions()闭环检测

bool LoopClosing::DetectAndReffineSim3FromLastKF

void LoopClosing::CorrectLoop()回环矫正

void LoopClosing::MergeLocal2()惯性模式下的地图融合

前言：

我的环境：ubuntu20、opencv4.5.5、eigen3.4、boodt1.75.python3.8硬件环境：5900hx、3060laptop

nvidia显卡驱动不是很好装、每次装完后都黑屏，按照网上解决办法试了没成功，最后用的社区开源驱动。因为个人原因选择了ubuntu20，但实验室老哥用的ubuntu18遇到的问题比我少很多，虽然官方测试了18与20，但还是推荐ubuntu18.

---------------------------------------------------------------------------------------------------------------------------------

安装教程：
这个老哥写的不错，我后面的重装基本按照这篇来的。其中的遇到的问题如下：
Ubuntu20.04 —— 新系统从头安装ORB-SLAM3过程（2022年）_@曾记否的博客-CSDN博客遇到
写在前面：本来是想在Ubuntu18.04上跑ORB-SLAM3的，但是不知道是那一步错了，在编译的最后一步出了好多错误，网上找了好多解决办法都不行，因为以前在Ubuntu18.04上跑了高博的slambook2的例程，安装了多个版本的库自己感觉是不同的库之间相互影响，索性直接安装Ubuntu20.04从头开始搭建运行环境，Ubuntu18.04上的错误以后再想办法。环境说明：虚拟机VMware + Ubuntu20.04 + Win10(i5-8500)准备工作：终端与主机复制粘贴sudo a
https://blog.csdn.net/qq_38364548/article/details/122220493?spm=1001.2014.3001.5506

 遇到和解决的问题
1.eigen版本太高了会有很多warning，在jetson nx或者树莓派上会低内存警告然后卡死。推荐3.2.0。

2.新版本的代码中没有euroc_examples.sh，我当时以为没编译成功。旧版本的推荐0.4beta或者b站up 计算机视觉life的，他的代码中有中文注解。但其实里面英文不难，挺好理解的

3.新版本代码需要c++14。如报错

make[2]: *** [CMakeFiles/ORB_SLAM3.dir/build.make:375：CMakeFiles/ORB_SLAM3.dir/src/MLPnPsolver.cpp.o] 错误 1
 
make[2]: *** [CMakeFiles/ORB_SLAM3.dir/build.make:76：CMakeFiles/ORB_SLAM3.dir/src/Tracking.cc.o] 错误 1
 
make[2]: *** [CMakeFiles/ORB_SLAM3.dir/build.make:245：CMakeFiles/ORB_SLAM3.dir/src/Frame.cc.o] 错误 1
 
make[2]: *** [CMakeFiles/ORB_SLAM3.dir/build.make:102：CMakeFiles/ORB_SLAM3.dir/src/LoopClosing.cc.o] 错误 1
 
make[2]: *** [CMakeFiles/ORB_SLAM3.dir/build.make:323：CMakeFiles/ORB_SLAM3.dir/src/G2oTypes.cc.o] 错误 1
 
make[2]: *** [CMakeFiles/ORB_SLAM3.dir/build.make:232：CMakeFiles/ORB_SLAM3.dir/src/Optimizer.cc.o] 错误 1
 
make[1]: *** [CMakeFiles/Makefile2:390：CMakeFiles/ORB_SLAM3.dir/all] 错误 2
 
make: *** [Makefile:84：all] 错误 2
 需要在orb-slam的cmakelist和orb-slam/expamples/ros中的cmakelist中都加入

add_compile_options(-std=c++14)
5.ORB_SLAM的ros包如果放在工作空间中不用加入ros_pakeage_path。老提示路径不对。是我卡了最久的问题。

6.第一次用opencv4，opencv4不自动生成opencv.pc，但可以设置开启生成。若用opencv4 ，则ros中的cmakelist需要更改。

7

-- Could NOT find PY_em (missing: PY_EM) 
CMake Error at cmake/empy.cmake:30 (message):
Unable to find either executable 'empy' or Python module 'em'...  try
installing the package 'python-empy'
 解决方法：

pip install empy
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

参考资料：
1：5小时让你假装大概看懂ORB-SLAM2源码_哔哩哔哩_bilibili

2：ORBSLAM源码及其算法详解 - 知乎

3.【泡泡机器人公开课】第三十六课：ORB-SLAM2源码详解-吴博_哔哩哔哩_bilibili

4.ROS：使用usb_cam软件包调试usb摄像头_通哈膨胀哈哈哈的博客-CSDN博客_usb_cam

5.ORB-SLAM2代码整理--LocalMapping线程_xiaoshuiyisheng的博客-CSDN博客_local mapping

6.这个老哥太强了：https://blog.csdn.net/ncepu_chen/category_9874746.htmlORB-SLAM2代码详解01: ORB-SLAM2代码运行流程_ncepu_Chen的博客-CSDN博客_orbslam2代码

7.回环部分：orb_slam代码解析(3)LocalMapping线程 - h_立青人韦 - 博客园 

前期准备：
以t265为例，基于ros运行时需要写一个setting，找一个Camera.type: "KannalaBrandt8"的yaml根据自己的相机改写就好，新版中有自带d435i和t265的yaml。在实际运行时候可能需要设置imu发布频率、话题名等。我的左摄像到imu一直提示none，用的同门的setting。

如果运行时候没有画面，需要查看入口函数，如ros_stereo_inertial.cc里订阅的话题名是否一样。并且t265、d435i、奥比中光等相机需要在launch文件中开启imu话题发布。

如果没有画面，显示waiting for images时，可以使用rqt_graph看看相机发布的图像、imu话题是否有被ros_setero_inertial订阅。

Kalibr相机校正工具安装与使用笔记

 笔记：
main函数
main函数 在Example里，我用的t265，后续准备用d435i，以t265为例。主函数在/Example/ROS/src/ros_stereo_inertial.cc

 【泡泡机器人公开课】第三十六课：ORB-SLAM2源码详解-吴博_哔哩哔哩_bilibili



int main(int argc, char **argv)
{
  ros::init(argc, argv, "Stereo_Inertial");// 
  ros::NodeHandle n("~");//初始化发布节点名和话题名 
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);//info级别的调试信息
  bool bEqual = false;//是否图像矫正
  if(argc < 4 || argc > 5)
  {
    cerr << endl << "Usage: rosrun ORB_SLAM3 Stereo_Inertial path_to_vocabulary path_to_settings do_rectify [do_equalize]" << endl;
    ros::shutdown();
    return 1;
  }//检查输入参数的个数
 
  std::string sbRect(argv[3]);//？
  if(argc==5)
  {
    std::string sbEqual(argv[4]);//？
    if(sbEqual == "true")
      bEqual = true;
  }
 
  // 初始化线程准备处理帧
  ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO,true);
 
  ImuGrabber imugb;
  ImageGrabber igb(&SLAM,&imugb,sbRect == "true",bEqual);
  
    if(igb.do_rectify)
    {      
       //yaml给到标定
        cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "ERROR: Wrong path to settings" << endl;
            return -1;
        }//读图
//内参、畸变
        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;//
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;
 
        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;
 
        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;
 
        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;
 
        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];
 
        if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
                rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }
 
        //计算无畸变和修正转换映射
cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,igb.M1l,igb.M2l);
        cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,igb.M1r,igb.M2r);
    }
 
  // Msubscriber左右目image imu 这里注意如果自己搭相机的要改setting中的话题名
  ros::Subscriber sub_imu = n.subscribe("/imu", 1000, &ImuGrabber::GrabImu, &imugb); 
  ros::Subscriber sub_img_left = n.subscribe("/camera/left/image_raw", 100, &ImageGrabber::GrabImageLeft,&igb);
  ros::Subscriber sub_img_right = n.subscribe("/camera/right/image_raw", 100, &ImageGrabber::GrabImageRight,&igb);
//同步imu和图像
  std::thread sync_thread(&ImageGrabber::SyncWithImu,&igb);
 
  ros::spin();
 
  return 0;
}
 
首先定义了图像和imu抓取器，获取imu和图像的sensor_时都用的互斥锁

 
 
class ImageGrabber//
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bRect, const bool bClahe): mpSLAM(pSLAM), mpImuGb(pImuGb), do_rectify(bRect), mbClahe(bClahe){}
 
    void GrabImageLeft(const sensor_msgs::ImageConstPtr& msg);//https://zhuanlan.zhihu.com/p/310285167
    void GrabImageRight(const sensor_msgs::ImageConstPtr& msg);//读取左右目
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);//ros图片转mat getimage
    void SyncWithImu();
 
    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    std::mutex mBufMutexLeft,mBufMutexRight;//定义互斥锁 用于左目右目
   
    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;
 
    const bool do_rectify;
    cv::Mat M1l,M2l,M1r,M2r;
 
    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};
 
orb_slam3:: system在system.h里，主要用于输入imu：初始化slam system，初始化包括定位、回环、查看线程。 首先会根据输入选择rgbd、mono等容器，然后初始化启动定位、关闭定位、换地图、reset地图等。

 
void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg)
{
  mBufMutexLeft.lock();
  if (!imgLeftBuf.empty())
    imgLeftBuf.pop();
  imgLeftBuf.push(img_msg);
  mBufMutexLeft.unlock();
}
//
锁，非空pop，空push，右目、imu同理


cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
 
  // ros图像转成cv::mat getimage
  cv_bridge::CvImageConstPtr cv_ptr;
void ImageGrabber::SyncWithImu()没咋看懂

void ImageGrabber::SyncWithImu()
{
  const double maxTimeDiff = 0.01;两目最大时间差
  while(1)
  {
    cv::Mat imLeft, imRight;
    double tImLeft = 0, tImRight = 0;
    if (!imgLeftBuf.empty()&&!imgRightBuf.empty()&&!mpImuGb->imuBuf.empty())//左右目图像imu非空
    {
      tImLeft = imgLeftBuf.front()->header.stamp.toSec();
      tImRight = imgRightBuf.front()->header.stamp.toSec();
 
      this->mBufMutexRight.lock();
      while((tImLeft-tImRight)>maxTimeDiff && imgRightBuf.size()>1)
      {
        imgRightBuf.pop();
        tImRight = imgRightBuf.front()->header.stamp.toSec();
      }
 
      this->mBufMutexRight.unlock();
 
      this->mBufMutexLeft.lock();
      while((tImRight-tImLeft)>maxTimeDiff && imgLeftBuf.size()>1)
      {
        imgLeftBuf.pop();
        tImLeft = imgLeftBuf.front()->header.stamp.toSec();
      }
      this->mBufMutexLeft.unlock();
 
      if((tImLeft-tImRight)>maxTimeDiff || (tImRight-tImLeft)>maxTimeDiff)
      {
        // std::cout << "big time difference" << std::endl;
        continue;
      }
      if(tImLeft>mpImuGb->imuBuf.back()->header.stamp.toSec())
          continue;
 
      this->mBufMutexLeft.lock();
 
//front是 element of the %queue.    timleft是左目第一帧给到stamp头，右目同理，左右目差最大时间差0.01时候，左目退到下一帧，直到同步。
      imLeft = GetImage(imgLeftBuf.front());
      imgLeftBuf.pop();
      this->mBufMutexLeft.unlock();
 
      this->mBufMutexRight.lock();
      imRight = GetImage(imgRightBuf.front());
      imgRightBuf.pop();
      this->mBufMutexRight.unlock();
//互斥锁读图片
      vector<ORB_SLAM3::IMU::Point> vImuMeas;
      mpImuGb->mBufMutex.lock();
      if(!mpImuGb->imuBuf.empty())//imu非空
      {
        // Load imu measurements from buffer
        vImuMeas.clear();
        while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec()<=tImLeft)//非空且时间戳同步
        {
          double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
          cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
          cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
          vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));//?
          mpImuGb->imuBuf.pop();
        }
      }
      mpImuGb->mBufMutex.unlock();
      if(mbClahe)
      {
        mClahe->apply(imLeft,imLeft);
        mClahe->apply(imRight,imRight);
      }
//自适应直方图均衡的东西 
      if(do_rectify)
      {
        cv::remap(imLeft,imLeft,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRight,M1r,M2r,cv::INTER_LINEAR);
      }//没找到remap在哪 资料说是图像矫正
 
      mpSLAM->TrackStereo(imLeft,imRight,tImLeft,vImuMeas);
//跟踪 开冲
      std::chrono::milliseconds tSleep(1);
      std::this_thread::sleep_for(tSleep);
    }
  }
}
System.cc
1读取当前传感器类型

2.读取settings

3.加载Vocabulary

4创建关键帧库

5创建多地图

6创建线程 跟踪、局部建图、回环、显示路径。

计算机视觉life的注释：



这里的参数、变量大都在system.h里定义，函数基本在对应功能的头文件，头文件的参数不在system里就在对应的cc中，有时候跳转定义不跳转。

 esensor传感器类型，枚举类。

mpViewer:The viewer draws the map and the current camera pose. It uses Pangolin.

Viewer:这里把mpviewer强制转换成空指针，Main thread function. Draw points, keyframes, the current camera pose and the last processed frame. Drawing is refreshed according to the camera fps. We use Pangolin.

接下来选择传感器类型，读setting文件、加载ORB_Vocabulary，这里加载Vocabulary调用了DBow2库（红色加粗标注了大体步骤）

    mpVocabulary = new ORBVocabulary();
    //新建Vocabulary
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);//读字典
//ORB vocabulary used for place recognition and feature matching.
 创建关键帧的DATABASE

pKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);
atlas地图：

{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{

atlas定义在atlas.h中,initkfid初始化时创建第一幅地图，mnLastInitKFidMap是在上一个地图最大关键帧的id +1（？mHasViewer没找到）

Atlas::Atlas(int initKFid): mnLastInitKFidMap(initKFid), mHasViewer(false)
{
    mpCurrentMap = static_cast<Map*>(NULL);
    CreateNewMap();
析构函数~atlas中创建一个迭代器，使用mspMaps存档当前地图信息，如果当前活跃地图有效，则存储当前地图为不活跃地图，如果当前地图非空或当前地图最新帧id大于当前地图第一帧id时，mnLastInitKFidMap = mpCurrentMap->GetMaxKFid()+1存储地图

  for(std::set<Map*>::iterator it = mspMaps.begin(), end = mspMaps.end(); it != end;)
 Map* pMi = *it;
 
        if(pMi)
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

创建atlas地图，初始化initKFid，接着判断是否需要imu信息来设置mbIsInertial以此开启追踪和预积分

 mpAtlas = new Atlas(0);
 
    if (mSensor==IMU_STEREO || mSensor==IMU_MONOCULAR)
 
        mpAtlas->SetInertialSensor();
   mpFrameDrawer = new FrameDrawer(mpAtlas);
    mpMapDrawer = new MapDrawer(mpAtlas, strSettingsFile);
FrameDrawer:

{{{{{{{{{{{{{{{{{

计算机视觉life的注释：
FrameDrawer::FrameDrawer(Atlas* pAtlas):both(false),mpAtlas(pAtlas)
{
    mState=Tracking::SYSTEM_NOT_READY;
    // 初始化图像显示画布
    // 包括：图像、特征点连线形成的轨迹（初始化时）、框（跟踪时的MapPoint）、圈（跟踪时的特征点）
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    mImRight = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}}}}}}}}}}}}}}}}}]

创建跟踪线程，

Tracking：

{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{

首先选择相机配置文件，得到相机输入，特征提取和匹配。

属实有点多，先搞system回头看吧。

}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

开启local mapping

localmapping：Local Mapper. It manages the local map and performs local bundle adjustment.

mptLocalMapping：The Tracking thread "lives" in the main execution thread that creates the System object.

判断三角测距是否可行，

 mpLocalMapper->mThFarPoints = fsSettings["thFarPoints"];
    if(mpLocalMapper->mThFarPoints!=0)
    {
        cout << "Discard points further than " << mpLocalMapper->mThFarPoints << " m from current camera" << endl;
        mpLocalMapper->mbFarPoints = true;
    }
    else
        mpLocalMapper->mbFarPoints = false;
创建闭环，创建可视化线程

 mpLoopCloser = new LoopClosing(mpAtlas, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR); // mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM3::LoopClosing::Run, mpLoopCloser);
    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
        mpLoopCloser->mpViewer = mpViewer;
        mpViewer->both = mpFrameDrawer->both;
    }
创建线程之间的指针

mpLocalMapper mpLoopCloser 地图和回环之间的指针

mpTracker mpLoopCloser 回环和跟踪

mpTracker mpLocalMapper 追踪和地图

刷志愿者时长去了 摸3天鱼刷志愿者时长去了 摸3天鱼刷志愿者时长去了 摸3天鱼

System::Track 以stereo

选择定位模式或者定位建图

 {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();// 请求停止线程
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }
 
            mpTracker->InformOnlyTracking(true);//只建图
            mbActivateLocalizationMode = false;设置为false避免上述操作循环
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }
  将imu存储到imudata中

if (mSensor == System::IMU_STEREO)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);
 得到图像深度时间戳文件名，记录mstate，当前的关键帧的mappoint，当前帧的关键点。

cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp,filename);
 
    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
 SaveTrajectoryTUM：

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }
 
    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();//所有关键帧
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);//帧排序
 
    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();、、第一帧的位姿
 
    ofstream f;
    f.open(filename.c_str());
    f << fixed;+
 
    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.
 
    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;
 
        KeyFrame* pKF = *lRit;
 
        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);
 
        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }
 
        Trw = Trw*pKF->GetPose()*Two;
 
        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);
 
        vector<float> q = Converter::toQuaternion(Rwc);
 
        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    // cout << endl << "trajectory saved!" << endl;
}
清明假期 开卷

上面大概走了下程序流程，现在打算走一遍特征点法→定位→建图



ORBextrsvtor.cc
原理：



以某个像素为中心p，取周围16个像素点，近似一个圆。设置一个阈值T，假设这个点附近有N个点的亮度大于或者小于p的亮度+T或者p的亮度-T则认为这个点是特征点。然后算这个圆的灰度质心。链接这个圆的几何中心与质心得到一个向量，这个向量就是fast关键点的方向。 

rbrief描述子旋转不变性
brief描述子：


先对图像附近进行高斯滤波减少噪声，然后在窗口内随机取两个点比较像素的大小，因为是以0、1表示的， 所以旋转后会变，比如原来0的现在是1



 rbrief描述子因为fast关键点有角度，所以计算描述子时以fast关键点的方向进行旋转，故rbeief具有旋转不变性 。

特征匹配
 SearchForInitializtion

orb特征提取使用四叉树特征提取比较均匀。特征匹配时在金字塔最底层的图像上进行，以特征点为中心半径一百的方形区域内遍历所有特征点，并计算他们描述子之间的汉明距离（二进制之间的距离）。最优距离小于50，计算最优距离和次优距离之间的比值。之后统计原特征点和匹配到的特征点之间的角度（每30为一个范围列直方图），判断特征点最多的三个方向。此时第二多的特征数量＜0.1*最多的方向，则证明第一方向为主方向。若第三多的方向＜0.1*最多的方向，则证明第一个第二多的方向为主方向。

SearchByBoW 2种

每个节点中包含很多帧 ，对每个节点中对应的特征点一一匹配

SearchByProjection 4种

利用EPnP计算位姿进行motion-onl-BA优化，如果内点不足则投影匹配

computeSim3 将回环关键帧的共视关系投影到回环关键帧中搜索匹配点，匹配点足够多，则说明回环成功

TrackWithModel

三帧中，某点从第一帧到第二帧之间的变换为T，则假设第二帧到第三帧的变换为T'=T，在第三帧点的区域附近进项匹配，找到汉明距离最小的点，若小于设置的阈值，则匹配成功；若没找到。则提高阈值重试

SearchForTriangulation

跟踪线程中没有跟踪到的特征点在参考关键帧中进行三角测距

Fuse

回环之后将位姿和地图点矫正

SearchBySim3

高斯图像金字塔：图像处理中的高斯金字塔和拉普拉斯金字塔_熊彬程的博客的博客-CSDN博客_高斯金字塔的作用

高斯金字塔(Gaussianpyramid): 用来向下采样，主要的图像金字塔
拉普拉斯金字塔(Laplacianpyramid): 用来从金字塔低层图像重建上层未采样图像，在数字图像处理中也即是预测残差，可以对图像进行最大程度的还原，配合高斯金字塔一起使用。
static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
首先计算特征点的方向，使用了几何中心和灰度质心的连线方向 

const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));
灰度值指针

定义 m_01为的加权，m_10为的加权，首先得到特征点所在图像块的灰度值指针center，除对称轴外每次可以计算(x,y)(x,-y),计算上方灰度值。轴上坐标加权m_10 += u * (val_plus + val_minus);在这一行上的和按照坐标加权。其中使用了fastAtan

 return fastAtan2((float)m_01, (float)m_10);
计算描述子的角度

static void computeOrbDescriptor(const KeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc)
const float factorPI = (float)(CV_PI/180.f);//一度对应的弧度大小
const float factorPI = (float)(CV_PI/180.f);//得到KPT特征点的对应角度
    for (int i = 0; i < 32; ++i, pattern += 16)
    {
		
        int t0, 	
			t1,		
			val;	
		
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;							//描述子本字节的bit0
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;					//描述子本字节的bit1
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;					//描述子本字节的bit2
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;					//描述子本字节的bit3
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;					//描述子本字节的bit4
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;					//描述子本字节的bit5
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;					//描述子本字节的bit6
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;					//描述子本字节的bit7
 
  
        desc[i] = (uchar)val;
每个描述子是一个二进制字符，有两个灰度值比较得来，一共比32次，每次比较除8bit信息 ，所以是256位比特。使用预先定义好的随机点集

 for(int i=1; i<nlevels; i++)  
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }
resizeCV13 图像分辨率操作（图像金字塔与resize()函数）_什么都只会一点的博客-CSDN博客

计算每层缩放系数的平方，其中nLevels:ORBextractor.nLevels: 8写死在yaml中，又用mnScaleLevels = mpORBextractorLeft->GetLevels();在Frame.cc中得到，在ORBextractor中返回。可以发现金字塔中第0层的尺度因子是1,然后每向上高一层，图像的尺度因子是在上一层图像的尺度因子1 1*1.2 1*1.2*1.2 1*1.2*1.2*1.2 ...

loat nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 (float)pow((double)factor, (double)nlevels));	//每个单位缩放系数所希望的特征点个数
for( int level = 0; level < nlevels-1; level++ )
    {
		//分配 cvRound : 返回个参数最接近的整数值
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
		//累计
        sumFeatures += mnFeaturesPerLevel[level];
		//乘系数
        nDesiredFeaturesPerScale *= factor;
    }计算除了最顶层外每层系数，每层根据采样缩放系数的倒数得到期望个数。希望个数减去每层个数和剩下的分配到最高层
 计算四分之一愿的边界，利用对称性得到边界

void ExtractorNode::DivideNode
将特征区域分为左上左下右上右下四块。存储边界点并将特征点放入对应的块中

 const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);
//特征区域取得中间坐标将区域分成四块，
 n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);//将四块坐标存储
 
 
 for(size_t i=0;i<vKeys.size();i++)//遍历特征点
 const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);将当前提取其中的特征点放入各自容器中
ORBextractor::DistributeOctTree
使用四叉树对一个金字塔图层中的特征点进行特征分发。

1.计算宽高、长宽比。resize提取器指针。
2.生成提取器节点

     ni.UL = cv::Point2i(hX*static_cast<float>(i),0);    //UpLeft
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);  //UpRight
		ni.BL = cv::Point2i(ni.UL.x,maxY-minY);		        //BottomLeft
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);             //BottomRight
    // S 将特征点分配到子提取器节点中
    for(size_t i=0;i<vToDistributeKeys.size();i++)
//遍历此提取器节点列表，标记那些不可再分裂的节点，删除那些没有分配到特征点的节点
  list<ExtractorNode>::iterator lit = lNodes.begin();
    while(lit!=lNodes.end())
为提取器的区域分配一个提取器指针，并分配每个区域特征点期望个数。如果个数为1 ，则标志标志位表述不可再分，个数为0则删除。

    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;//记录了可以继续分裂的节点
将一个初始化节点分为4个。利用四叉树划分区域。然后继续遍历提取器，如果提取器只有一个特征点，则list++，else则分成四个子区域。如果四个子区域中特征点数>0，则把这个子区域的提取器添加到列表前面。添加这个子区域中特征点的数目和节点指针vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));之后再循环分配器、提取器。分裂方式是后加的先分裂，先加的后分裂。

   if((int)lNodes.size()>=N|| (int)lNodes.size()==prevSize)
如果当前特征点个数达到了阈值或每个区域已经不能再分则停止
  else if(((int)lNodes.size()+nToExpand*3)>N)如果当前特征点和将要展开的点已经超过所要个数。，则想办法使其在达到或者刚超过所需的特征点数量时退出。一直循环到标志位被标志
此时我们使用pair将可以分裂的数量保存后用sort进行排序 ，从后往前遍历这个使用pair的vector对每个点进行分裂。pair的p2位待分裂节点、给到DivideNode（n1、n2、 3、 4）,将需要分裂的点放到前面，并且如果大于1，则放入lnode的前面

 vector<cv::KeyPoint> vResultKeys;//保存将兴趣点
for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)//遍历
   for(size_t k=1;k<vNodeKeys.size();k++)
遍历容器中的特征点，得到第一个点的最大响应值作为后续的最大响应值
         if(vNodeKeys[k].response>maxResponse)
若最大响应值大于第一个最大响应值则更新最大响应值
 vResultKeys.push_back(*pKP);
将这个区域的最大响应值加入最终结果容器
 ORBextractor::ComputeKeyPointsOctTree计算四叉树的特征点，第一层vector时当前图层的所有特征点，第二层是整个图像金字塔中的所有特征点。遍历图层

首先计算图像坐标边界，存储需要进行分配的点和特增点个数。开始遍历，FAST提取FAST角点

FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),	
 if(vKeysCell.empty())
                {
					//那么就使用更低的阈值来进行重新检测
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),	//待检测的图像
 for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {
						//NOTICE 到目前为止，这些角点的坐标都是基于图像cell的，现在我们要先将其恢复到当前的【坐标边界】下的坐标
						//这样做是因为在下面使用八叉树法整理特征点的时候将会使用得到这个坐标
						//在后面将会被继续转换成为在当前图层的扩充图像坐标系下的坐标
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
 

cv::pointOpencv KeyPoint类_DXT00的博客-CSDN博客

ORBextractor::ComputeKeyPointsOld/
使用老办法来得到特征点

计算cell个数，保存cell特征点数、位置、是否只有一个特征点等。之后遍历网格。如果cell在第一行则计算初始边界，如果在最后一个则计算增量坐标（有几个像素用几个，如果太小则舍弃），

Mat cellImage = mvImagePyramid[level].rowRange(iniY,iniY+hY).colRange(iniX,iniX+hX);
cellimage给他扣出来，第几层.rosrange.lowrange
如果提取出的特征点小于3个，清空后降低阈值重新提取。如果满足，则保存特征点数目
根据评分使特征点分布均匀。根据响应值保留符合要求的特征点，并且进行坐标的转换

 KeyPointsFilter::retainBest(keysCell,			//输入输出，用于提供待保留的特征点vector，操作完成后将保留的特征点存放在里面
																//其实就是把那些响应值低的点从vector中给删除了
											nToRetain[i][j]);	//指定要保留的特征点数目
 
 
 
  for(size_t k=0, kend=keysCell.size(); k<kend; k++)//坐标变换
ORBextractor::operator()计算特征点

1.检查是否非空

2，构建图像金字塔ComputePyramid(image);

3.可以使用四叉树或者传统方法进行特征点计算，使特征点均匀化。

4.深拷贝描述子到新矩阵descriptors

        4.1统计金字塔每层特征点个数并累加求和。

        4.2存储图像金字塔中描述子的矩阵

5.使用高斯模糊避免噪声影响描述子计算

6.计算高斯模糊之后的描述子

7.对高斯模糊后更层上的描述子恢复到原本的图像金字塔坐标上

   void ORBextractor::ComputePyramid(cv::Mat image)
MapPoint 
‘这部分函数命名很舒服，基本看下名称就知道干啥的，就是变量名有点烦

MapPoint::SetWorldPos 设置地图点坐标

cv::Mat MapPoint::GetWorldPos()返回世界坐标

cv::Mat MapPoint::GetNormal()获取平均观测方向

void MapPoint::AddObservation添加观测点

std::map<KeyFrame*, std::tuple<int,int>> MapPoint::GetObservations()能够观测到当前地图点的所有关键帧及该地图点在KF中的索引

void MapPoint::SetBadFlag()先标记后删除，遍历mObservations，标记后直接擦除内存mpMap->EraseMapPoint(this);

void MapPoint::Replace(MapPoint* pMP)替换地图点
同一个点跳过 ，pMP用来替换的那个地图点 ，观测到的点的指针给到obs并清除原有观测

 map<KeyFrame*,tuple<int,int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
    for(map<KeyFrame*,tuple<int,int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
 将原有能观测到原来地图点的关键帧复制到要替换的地图点上
 
不在则替换  if(!pMP->IsInKeyFrame(pKF))
        {
            // 如果不在，替换特征点与mp的匹配关系
            if(leftIndex != -1){
                pKF->ReplaceMapPointMatch(leftIndex, pMP);
                pMP->AddObservation(pKF,leftIndex);
            }
            if(rightIndex != -1){
                pKF->ReplaceMapPointMatch(rightIndex, pMP);
                pMP->AddObservation(pKF,rightIndex);
            }
如果在 删除旧的 新的不动
更新原有 可观测此点的帧数 可视点（非特征点和地图点 能够看到的点） 描述子
void MapPoint::ComputeDistinctiveDescriptors()
* 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要判断是否更新当前点的最适合的描述子。先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值

1.跳过坏点、

2.遍历点获取描述子放入vDESCRIPTORS

.3.遍历点取得距离给distij

 int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
4. 选择最有代表性的描述子，它与其他描述子应该具有最小的距离中值，距离中值为INT_MAX

vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        // 获得中值
        int median = vDists[0.5*(N-1)];
 
        // 寻找最小的中值
        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
返回描述子，确定是否在关键帧中并得到这个点在关键帧中对应的id

void MapPoint::UpdateNormalAndDepth()
更新平均观测方向以及观测距离范围。由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要更新相应变量。创建新的关键帧的时候会调用。

1.获得该地图点的相关信息

 observations=mObservations; // 获得观测到该地图点的所有关键帧
        pRefKF=mpRefKF;             // 观测到该点的参考关键帧（第一次创建时的关键帧）
        Pos = mWorldPos.clone();    // 地图点在世界坐标系中的位置
2.能观测到此点的所有观测方向归一化。得到该点的朝向

主要是用该点的相机坐标减去其世界坐标，normal为法线。首先得到法线。然后计算改点到参考关键帧相机的距离得到这个点的观测距离上下限和地图平均观测方向

float MapPoint::GetMaxDistanceInvariance()
这里没太看懂



// 在进行投影匹配的时候会给定特征点的搜索范围,考虑到处于不同尺度(也就是距离相机远近,位于图像金字塔中不同图层)的特征点受到相机旋转的影响不同,
// 因此会希望距离相机近的点的搜索范围更大一点,距离相机更远的点的搜索范围更小一点,所以要在这里,根据点到关键帧/帧的距离来估计它在当前的关键帧/帧中,
// 会大概处于哪个尺度
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        // mfMaxDistance = ref_dist*levelScaleFactor 为参考帧考虑上尺度后的距离
        // ratio = mfMaxDistance/currentDist = ref_dist/cur_dist
        ratio = mfMaxDistance/currentDist;
    }
 
    // 同时取log线性化
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;
 
    return nScale;
}
 
/**
 * @brief 根据地图点到光心的距离来预测一个类似特征金字塔的尺度
 * 
 * @param[in] currentDist       地图点到光心的距离
 * @param[in] pF                当前帧
 * @return int                  尺度
 */
int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }
 
    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;
 
    return nScale;
}
 
Map* MapPoint::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}
 
void MapPoint::UpdateMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}
 
} //namespace ORB_SLAM
Frame.cc
麻了呀写完了都，结果发布之后就没了

Frame::Frame
构造frame

for(int i=0;i<FRAME_GRID_COLS;i++)

for(int j=0; j<FRAME_GRID_ROWS; j++){

mGrid[i][j]=frame.mGrid[i][j];

if(frame.Nleft > 0){

mGridRight[i][j] = frame.mGridRight[i][j];

}

得到位姿和线速度

Frame::Frame
1.帧id自加

2.得图像金字塔的参数，如层数、层间比等

3.左右目提取orb特征点，开了两个线程。得到特征点的个数

4.opencv矫正

5.双目之间特征匹配。去畸变，计算图像边界。 这边没看懂怎么进行匹配的

6.特征点分配到网格中

void Frame::AssignFeaturesToGrid()
将特征点分配到网格中

1.遍历行列。有特征点则分配空间

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
2.遍历特征点，放入特征点的索引值。用了两个( ? : :)

  const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];
void Frame::ExtractORB
提取特征点

     monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
    else
        monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
判断地图点是否在帧中

// mbTrackInView是决定一个地图点是否进行重投影的标志
    	// 这个标志的确定要经过多个函数的确定，isInFrustum()只是其中的一个验证关卡。这里默认设置为否
        pMP->mbTrackInView = false;
1.得到当前地图点的世界坐标，并且转换成相机坐标下的三维点。

2.检查这个地图点的深度

3.投影地图点并检测是否在图像有效范围内

        if(uv.x<mnMinX || uv.x>mnMaxX)
            return false;
        if(uv.y<mnMinY || uv.y>mnMaxY)
            return false;
若有效则赋给追踪坐标

const int nPredictedLevel = pMP->PredictScale(dist,this);（图像金字塔中的预测尺度），之后保存得到的尺度

bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)没怎么看懂

大致是如果地图点在帧中可以被挂测到填充追踪的东西？

vector<size_t> Frame::GetFeaturesInArea
获取半径为r的圆域内的特征点索引列表。

检查是否左右、上下边界符合。遍历网格，如果格子里有特征点，则检查索引，判断是否在minlevel和maxlevel中，判断是否在圆形区域中。设置为候选特征点。factorX = r。factorY = r。

(x-mnMinX-r)=x-mnMinX-factorX

可是这里不是用算了占了多少网格吗？为什么可以得到在哪个具体网格？

//更新 是算了左边界到图像的网格数。再和左边界比  判断 是否越界。上下左右同理

    // (mnMaxX-mnMinX)/FRAME_GRID_COLS：表示列方向每个网格可以平均分得几个像素（肯定大于1）
    // mfGridElementWidthInv=FRAME_GRID_COLS/(mnMaxX-mnMinX) 是上面倒数，表示每个像素可以均分几个网格列（肯定小于1）
	// (x-mnMinX-r)，可以看做是从图像的左边界mnMinX到半径r的圆的左边界区域占的像素列数
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
				// 根据索引先读取这个特征点 
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
 
 
 
     const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;圆心坐标减特征点的坐标
bool Frame::PosInGrid得到特征点在网格中的坐标

void Frame::ComputeBoW()计算当前帧的词袋 将mDescriptors转为DBOW要求的格式，转换函数在converter.cc中

void Frame::UndistortKeyPoints()

特征点去畸变，去畸变后的特征点存在mvKeysUh中，只有单目需要去畸变，所以代码中为pinhole*。双目不需要矫正

1.判断是否需要矫正

2.使用opencv进行矫正。遍历每个特征点，并将坐标保存在矩阵中。为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y）

   mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);
3。遍历后覆盖存储。

void Frame::ComputeImageBounds(const cv::Mat &imLeft)计算图像边界，校正后的四点边界 不能围成一个矩形，故以四点连线为边框。

void Frame::ComputeStereoMatches()
双目匹配

1.分配内存存储右图匹配点索引，深度信息存储。设置orb相似度值域。从第0层开始遍历金字塔

2.遍历金字塔，得到特征点的行号。特征点不一定是在唯一一层的。计算特征点在行方向上可能存在的像素偏移。并保存可能在哪行上vRowIndices

3.保存sad块匹配相似度和左图特征点索引，搜索左图相似的右图特征点。计算理论上的最佳搜索范围，初始化最佳相似度，用最大相似度，以及最佳匹配点索引

3.1粗配准. 左图特征点il与右图中的可能的匹配点进行逐个比较,得到最相似匹配点的相似度和索引

 for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];
 
            // 左图特征点il与带匹配点ic的空间尺度差超过2，放弃
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;
 
            // 使用列坐标(x)进行匹配，和stereomatch一样
            const float &uR = kpR.pt.x;
 
            // 超出理论搜索范围[minU, maxU]，可能是误匹配，放弃
            if(uR>=minU && uR<=maxU)
            {
                // 计算匹配点il和待匹配点ic的相似度dist
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);
 
				//统计最小相似度及其对应的列坐标(x)
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
3.2精确匹配 若匹配到的右图特征点的最佳描述子小于设定值域。计算右图特征点列坐标和金字塔尺度。这一步是为了得到右图特征点的坐标。提取左图中以特征点为中心，半径为w的图像块

 cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
开始滑动窗口匹配

or(int incR=-L; incR<=+L; incR++)
            {
                // 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
 
                // sad 计算
                float dist = cv::norm(IL,IR,cv::NORM_L1);
                // 统计最小sad和偏移量
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }
 
                //L+incR 为refine后的匹配点列坐标(x)
                vDists[L+incR] = dist;
            }
亚像素拟合抛物线寻找最佳匹配位置。这部分直接就没看懂。过了。需要的时候再翻

4删除outliers 因为sad滑动匹配时会受到光照等噪声的干扰造成误匹配，

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;
 
    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
 void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)计算立体深度信息。获取矫正前的特征点坐标取得深度图

cv::Mat Frame::UnprojectStereo(const int &i)反投影到世界坐标上

Frame::Frame 
左右特征提取，得到总特征数。初始化并去畸变。计算一些参数，基线，右目图像网格计算。图像拼接，计算鱼眼相机特征点匹配。描述子并集。分配到网格，特征点去畸变

void Frame::ComputeStereoFishEyeMatches()鱼眼相机特征匹配

1左右目特征点遍历、得到描述子

2.左右目暴力匹配（这里的2没查到是什么意思

3.遍历匹配，对于好的匹配检查视差和重投影错误以丢弃虚假匹配

bool Frame::isInFrustumChecks 没太看懂  主要是检查地图点是否在帧内，包括检查距离、角度等

ORB-SLAM3 IMU(李群)+Frame+KeyFrame+MapPoint_xiaoma_bk的博客-CSDN博客

KeyFrame
KeyFrame::KeyFrame构造keyframe

初始化用于加速的网格对象mnGridCols，遍历行列复制原有网格中的特征点索引

然后得到相机坐标、imu速度等

void KeyFrame::ComputeBoW()匹配描述子向量

void KeyFrame::SetPose(const cv::Mat &Tcw_)设置当前关键帧的位姿

得到一些参数 imu 位姿 位姿逆

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
为关键帧创建链接

 if(!mConnectedKeyFrameWeights.count(pKF)) // count函数返回0，mConnectedKeyFrameWeights中没有pKF，之前没有连接
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight) // 之前连接的权重不一样，更新
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }
 
    // 如果添加了更新的连接关系就要更新一下,主要是重新进行排序
    UpdateBestCovisibles();
mConnectedKeyFrameWeights是一个std::map,无序地保存当前关键帧的共视关键帧及权重。size_type count (const key_type& k) const;
Count elements with a specific key - 返回匹配特定键的元素数量

void KeyFrame::UpdateBestCovisibles()
 按照权重对连接的关键帧进行排序.迭代器遍历共视图

-》second value ->first key

for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));
 sort(vPairs.begin(),vPairs.end());
进行排序
定义帧和权重的列表
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        if(!vPairs[i].second->isBad())
        {
			// push_front 后变成从大到小
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
    }
 
//遍历删除坏点 并且使用push front把权重倒置
set<KeyFrame*> KeyFrame::GetConnectedKeyFrames() 得到与当前关键帧关联的15个关键帧，迭代器遍历关键帧。

 vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()得到用权值排序的关键帧

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N) 得到N个用权值排序的关键帧

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)得到与该关键帧连接的权重大于等于w的关键帧

set<MapPoint*> KeyFrame::GetMapPoints()获取地图点  给到pMP

int KeyFrame::TrackedMapPoints(const int &minObs) 统计 但当前帧中地图点的个数

void KeyFrame::UpdateConnections(bool upParent)
1.首先获得该关键帧的mappoint，然后统计这些点和其他帧的共视关系。

2.建立和别的关键帧的链接，他们的权重就是当前帧和其他关键帧中公示点的个数。

3.设置阈值，如果关键帧的个数小于阈值，则不保存这条边。只保留共视程度高的边。

4.根据帧之间边的权重构建最大生成树。

//这里坏点也继续了？

 for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {创建迭代期遍历当前帧的地图点
map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
  // 对于每一个MapPoint点，observations记录了可以观测到该MapPoint的所有关键帧
KFcounter该关键帧中看到的地图点计数，mit->first是地图看到的关键帧

      if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            // 对应权重需要大于阈值，对这些关键帧建立连接
            vPairs.push_back(make_pair(mit->second,mit->first));
            // 对方关键帧也要添加这个信息
            // 更新KFcounter中该关键帧的mConnectedKeyFrameWeights
            // 更新其它KeyFrame的mConnectedKeyFrameWeights，更新其它关键帧与当前帧的连接权重
            (mit->first)->AddConnection(this,mit->second);
如果没有超过阈值，则对权重最大的关键帧建立连接

        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
关键帧之间的顺序根据权重进行排序

    sort(vPairs.begin(),vPairs.end());    
 
        // 更新当前帧与其它关键帧的连接权重
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
//改变父关键帧为什么也要用add child

void KeyFrame::SetBadFlag()
有的帧虽然冗余，但是要用于回环或者sim3计算，所以当时不能删。但用完之后需要进行删除

 for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
    {
        mit->first->EraseConnection(this); // 让其它的KeyFrame删除与自己的联系
    }
这一步主要是需要处理好生成树中的父节点

1首先遍历需要删除的节点的子节点，并且处理了一些不能删的特殊情况，初始关键帧不能删除

2.遍历子关键帧中每一个与他共视的帧。找当前帧的父关键帧与其子节点之间是否存在共视关系。如过当前帧的子关键帧和当前帧的父关键帧之间共视。则当前帧的子关键帧找到备选父节点。此时寻找并更新权值最大的那个共视关系。然后把当前帧的父节点更新为当前帧子节点的父节点

3.如果子节点没有找到父节点，则直接把自己的爷爷节点当做新的父节点。

void KeyFrame::EraseConnection(KeyFrame* pKF)删除共视关系 此时的判定条件是权值不为空

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r, const bool bRight) const

获取特征点附近的邻域特征点id ，使用ceil滑块

cv::Mat KeyFrame::UnprojectStereo(int i)平面点投影到相机坐标

float KeyFrame::ComputeSceneMedianDepth(const int q)弹幕摄像头评估距离。在当前帧下对所有地图点的深度进行从小到大

下面是一些坐标变幻的函数，例如右目位姿，转换等等

tracking
首先从配置文件中读取相机参数

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings) 根据相机类型配置参数

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)根据yaml设置参数

 nFeatures = node.operator int();
以整数形式返回节点内容。如果节点存储浮点数，则将其舍入 
 mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
创建特征提取器
cv::Mat Tracking::GrabImageStere

将左右目图片转为灰度图，根据不同传感器类型选择数据，开始跟踪。输出世界坐标系到该帧相机坐标系的变换矩阵。

void Tracking::PreintegrateIMU()对imu进行预积分

if(!mCurrentFrame.mpPrevFrame)上一帧不存在,说明两帧之间没有imu数据，不进行预积分

一开始没有数据，所以需要等一会，不需要挂起线程bool bSleep = false;

之后得到imu数据，以第一个imu数据为起始

  IMU::Point* m = &mlQueueImuData.front();
如果imu的时间戳和帧本身的时间差0.001则抛弃imu数据
// 得到两帧间的imu数据放入mvImuFromLastFrame中,得到后面预积分的处理数据
                    mvImuFromLastFrame.push_back(*m);
                    break;
第二步

m个imu数据只有m-1个预积分

const int n = mvImuFromLastFrame.size()-1;
    // 构造imu预处理器,并初始化标定数据
    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);
两帧之间求时间差在加加速度不变的假设上算出第一帧图像到第二个imu数据的中值积分

后面看不太懂了,打算过一下视频

void Tracking::MonocularInitialization()
单目初始化

1.单目初始帧特征点数需要大于100,更新上一帧。把当前帧更新变成上一帧并记录特征点。构造初始化器。如果融合了imu，则删除保存的上一帧的imu预积分，给参数和imu的bias更新上一帧的imu预积分

2如果当前帧特征点数不超过100，则重新构造初始器

3.初始帧和第二帧寻找匹配点

4.匹配的特征点数目不够则重新初始化

5.通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints

6 初始化成功后，删除那些无法进行三角化的匹配点

7.初始化的第一帧作为世界坐标系

8.创建初始化的单目世界地图

void Tracking::CreateInitialMapMonocular()
单目初始化成功后使用三角化的点生成地图点

1.将当前帧和上一帧设为关键帧，清空imu预积分，将这两真的关键点转为描述子，然后插入地图。

2.使用3d点构建地图，然后为该地图点添加对应的属性：观测到该地图点的关键帧，该地图点的描述子和观测方向与距离。

3.更新关键帧之间的连接

4.进行ba优化，优化所有位姿和三维点

初始化成功的两个条件：1平均深度大于0，在当前帧中被观测到的地图点数目应大于50

5.帧变换和3d点尺度也归一化到1

6.将关键帧插入局部地图，更新位姿和局部地图点

7.使用恒速模型估计

void Tracking::StereoInitialization()
双目相机初始化

1.判断特征点数目大于500，双目带imu的类型，当前帧与上一帧之间差0.5。判断之后将imu进行预积分

2.设置imu方向和速度，若不带imu设置相机坐标

3.当前帧构造关键帧，初始地图中添加关键帧，添加地图点，

// 为该MapPoint添加属性：

// a.观测到该MapPoint的关键帧

// b.该MapPoint的描述子

// c.该MapPoint的平均观测方向和深度范围

4..当前帧更新为上一帧，

void Tracking::Track() 
1初始化地图 ，判断是否是badimu 得到alas地图的currentframe

2.处理异常时间帧，若没使用有imu补偿，则重置地图，若imu模式且初始化完成且进行了ba优化则创建新的子图，若完成了初始化但没进行ba优化则重置active地图

3.imu模式且有上一帧的情况下设置imu偏移，若没上一帧则进行预积分

4.判断地图是否被更新

5.初始化，根据相机类型和带不带imu进行初始化

6.开始跟踪

6.1slam模式

1.检查上一帧被替换的地图点

2.如果velocity空且为初始化或者当前帧和最新帧差了两帧 使用参考帧进行跟踪，参考帧跟踪是将普通帧的描述子转化成bow向量，通过bow加速当前帧和参考帧之间的匹配，将上一帧的姿态作为当前帧的初始位姿，并且重投影到3d世界，剔除优化后的匹配点的外点。否则就用恒速模型。

如果当前帧距离上次重定位成功不到1s或者单目+IMU 或者 双目+IMU模式，标记为LOST。若当前地图关键帧大于10帧且距离上次重定位不超过1s，标记为RECENTLY_LOST

3.异常情况下：if (mState == RECENTLY_LOST)，且有imu初始化，则使用imu预测位姿，如果当前imu模式下丢帧超过5s还没找回，设置为lost。else进行纯视觉重定位，主要是bow搜索和EPNP求解位姿，视觉重定位失败设为lost

4.如果lost状态，当前地图中关键帧数目小于10重置地图，大于10创建新地图

6.2纯定位模式

1.如果状态为lost，Relocalization。mbV0表示此帧地图点收10个地图点，如果状态正常则根据是否有velocity来进行恒速模型或者关键帧跟踪。如果mbV0 ，且Velocity.empty则使用恒速模型进行位姿估计，如果没有运动模型则使用重定位方法

2.如果恒速模型成功，重定位失败，则使用之前暂存的结果，并且增加当前当前地图被观测到的次数。如果重定位成功整个跟踪过程正常进行

7.得到初始姿态后，对local map进行追踪得到更多的匹配，并且优化当前位姿，

bok只要局部地图跟踪成功就可以，可以是上一帧和当前帧更新成功，也可以是上一帧跟踪失败但是重定位成功。而mstate只要第一步上一帧跟踪成功就为ok

    if(bOK)
        mState = OK;
如果带imu，但没有初始化或者进行ba优化则重置地图。或者直接置为lost

如果当前帧距离上次重定位帧超过1s，用当前帧时间戳更新lost帧时间戳

如果刚刚发生重定位并且IMU已经初始化，则保存当前帧信息，重置IMU

8.if(bOK || mState==RECENTLY_LOST 更新恒速模型

9.遍历当前帧清除观测不到的地图点

10Step 9.3 清除恒速模型跟踪中 UpdateLastFrame中为当前帧临时添加的MapPoints（仅双目和rgbd）

11.bok=true或者recently_lost时，在bnnedkf=true情况下判断是否需要插入关键帧

12删除在ba检测中检测为外点的点

如果mstate标志位lost，如果地图关键帧小于5，且没有初始化，则重置当前地图并退出当前跟踪。如果已经初始化，则创建新地图。设置参考帧，并且保存当前帧编程上一帧。

13.记录位姿

相机初始化之后为预测位姿，有三种方法：恒速模型、参考帧、重定位

1.恒速模型估计位姿
1.bool Tracking::TrackWithMotionModel() 

// 最小距离 < 0.9*次小距离 匹配成功，检查旋转

2.ORBmatcher matcher(0.9,true);这里没看懂，之后更新上一帧位姿 。如果mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU则使用imu进行位姿估计。否则使用恒速模型得到当前帧位姿

3.设置搜索半径。使用上一帧的地图点投影匹配，匹配点小于20扩大匹配点再来一次。

4.利用3d-2d进行位姿优化，

5.删除地图中外点

2.参考帧估计位姿
TrackReferenceKeyFrame()

1.将当前帧的描述子更新为bow向量

2.使用bow加速当前帧与参考帧之间的特征匹配。若匹配数目小于15则跟踪失败。

3.将上一帧的位姿作为当前帧位姿，3d-2d投影优化位姿

4.清楚外点

3.重定位估计位姿
 Relocalization()

1.计算特征点的bow映射，找到与当前帧相似的关键帧组

2.遍历关键帧进行bow快速匹配。如果匹配数小于15则放弃这个关键帧。如果匹配数目够则初始化mlpnpsolver

3.遍历当前候选关键帧，通过MLPnP算法估计姿态，迭代5次。计算出的内点进行ba优化。遍历所有的内点进行ba优化。内点剩10个以内跳过当前帧

4.如果投影点较少，则使用3d-2d投影。如果过投影后匹配较多特征点对，则再次使用3d-2d pnp ba优化。如果匹配的特征点＞30，＜50，则缩小滑动窗口进行搜索。如果搜到50个以上则达到要求使用ba一下。

bool Tracking::TrackLocalMap()
1.UpdateLocalMap();更新局部底部，包括更新局部关键帧和局部地图点。

2.SearchLocalPoints();筛选局部地图中新增的在视野中的点，投影到当前帧搜索匹配，得到更多的匹配关系

3.ba优化位姿，imu未初始化仅优化位姿，若距离上次重定位比较近则不进行优化

4.遍历帧中关键点，当前帧的地图点可以被当前帧观测到其被观测统计量加1。nObs： 被观测到的相机数目，单目+1，双目或RGB-D则+2

5.根据跟踪匹配数目及重定位情况决定是否跟踪成功

bool Tracking::NeedNewKeyFrame()
判断是否需要创建关键帧

1.imu模式且未完成初始化，并且距离上一帧时间超过0.25插入关键帧。

2.纯VO模式下不插入关键帧。如果局部地图线程被闭环检测使用，则不插入关键帧

3.如果距离上一次重定位比较近，并且关键帧数目超出最大限制，不插入关键帧

4.得到参考关键帧跟踪到的地图点数量，查询局部地图线程是否繁忙，当前能否接受新的关键帧，对于双目或RGBD摄像头，统计成功跟踪的近点的数量，如果跟踪到的近点太少，没有跟踪到的近点较多，可以插入关键帧。

5.设定比例阈值，当前帧和参考关键帧跟踪到点的比例，比例越大，越倾向于增加关键帧

很长时间没有插入关键帧，可以插入

const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;

6.满足插入关键帧的最小间隔并且localMapper处于空闲状态，可以插入

const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle);

7.在双目，RGB-D的情况下当前帧跟踪到的点比参考关键帧的0.25倍还少，或者满足bNeedToInsertClose

8.和参考帧相比当前跟踪到的点太少 或者满足bNeedToInsertClose；同时跟踪到的内点还不能太少

9.单目/双目+IMU模式下，并且IMU完成了初始化（隐藏条件），当前帧和上一关键帧之间时间超过0.5秒

10.单目+IMU模式下，当前帧匹配内点数在15~75之间或者是RECENTLY_LOST状态

void Tracking::CreateNewKeyFrame()
将当前帧构造成关键帧，imu模式下对imu进行预积分

 KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    pKF->SetNewBias(mCurrentFrame.mImuBias);
	// Step 2：将当前关键帧设置为当前帧的参考关键帧
3.对于双目或rgbd摄像头，为当前帧生成新的地图点，得到有深度的点（不一定是地图点），排序后找出不是地图点的生成地图点。若此点在上一次针没有被观测到就生成一个临时的地图点，并得到这些点的属性，点的深度超过深度的阈值或数量超过100个停止。

4.插入关键帧，关键帧插入到列表 mlNewKeyFrames中，等待local mapping。

LocalMapping
主函数void LocalMapping::Run()
1. 润函数里有个标志位mbfinished，false时表示正在运行，之后设了一个主循环告诉不要插入关键帧

2. 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等

3 根据地图点的观测情况剔除质量不好的地图点

4 当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳

			// 已经处理完队列中的最后的一个关键帧
            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                //  Step 5 检查并融合当前关键帧与相邻关键帧帧（两级相邻）中重复的地图点
                // 先完成相邻关键帧与当前关键帧的地图点的融合（在相邻关键帧中查找当前关键帧的地图点），
                // 再完成当前关键帧与相邻关键帧的地图点的融合（在当前关键帧中查找当前相邻关键帧的地图点）
                SearchInNeighbors();
 6.处理完最后一个帧，并且没有闭环检测请求停止localmapping

判断 若此时地图关键帧＞2

1. 处于IMU模式并且当前关键帧所在的地图已经完成IMU初始化 

计算上一关键帧到当前关键帧相机光心的距离 + 上上关键帧到上一关键帧相机光心的距离

2.当前关键帧所在的地图尚未完成IMU BA2

如果累计时间差小于10s 并且 距离小于2厘米，认为运动幅度太小，不足以初始化IMU，将mbBadImu设置为true

否则 如果单目内点数大于75 或 多目内点数大于100 局部地图+IMU一起优化，优化关键帧位姿、地图点、IMU参数        

3.当前关键帧所在地图未完成IMU初始化（第一阶段）

7.imu初始化 检测并剔除当前帧相邻的关键帧中冗余的关键帧

判断条件 进行ba

8.将当前帧加入到闭环检测队列中

mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

当需要终止线程时，检查 isstop&& 帧队列中是否还有帧

 else if(Stop() && !mbBadImu) // 
        {
 
            while(isStopped() && !CheckFinish())
            {
                // cout << "LM: usleep if is stopped" << endl;
				// 如果还没有结束利索,那么等等它
                usleep(3000);
            }
            // 然后确定终止了就跳出这个线程的主循环
            if(CheckFinish())
                break;
跳出线程后 acceptkeyframe置为true，并且线程睡3秒

void LocalMapping::ProcessNewKeyFrame()处理关键帧
1. 取出帧

2.计算bow，更新地图中的描述子地图点等，遍历当前帧关键点



如果

 // 如果地图点不是来自当前帧的观测，是本帧匹配到的为当前地图点添加观测
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    // 获得该点的平均观测方向和观测距离范围
                    pMP->UpdateNormalAndDepth();
                    // 更新地图点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();
如果是本真跟踪到的且题图点中却没有包含这个地图点的关键帧的信息

mlpRecentAddedMapPoints.push_back(pMP); 
3.更新关键帧插入地图中

void LocalMapping::MapPointCulling()剔除不好的地图点
1.根据相机设置阈值 

2.遍历新添加的地图点，1）坏点直接删除，2）跟踪到该MapPoint的Frame数相比预计可观测到该MapPoint的Frame数的比例小于25%，删除。3）从该点建立开始，到现在已经过了不小于2个关键帧，但是观测到该点的关键帧数却不超过cnThObs帧，那么删除该点。从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点

void LocalMapping::CreateNewMapPoints()
1.设置nn作为最佳共视关系的数目。

2.在当前关键帧的共视关系中找到共视程度最嘎的nn帧相邻关键帧

//?但是这里nn不是一个数字吗 

//如果达不到nn就把关键帧返回

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
 
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);
 
}
 vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
3，imu模式下如果相邻共视关键帧小于nn则添加附近的帧

4.遍历相邻关键帧，如果特征点距离小于基线不生成地图点，如果特别远是基线的100倍不生成地图点 ratioBaselineDepth= baseline/medianDepthKF2;

5.根据两帧之间的位姿求变换矩阵，通过BoW对两关键帧的未匹配的特征点快速匹配，用极线约束抑制离群点，生成新的匹配点对

6.利用特征点。生成地图点，利用匹配点投影得到视差的余弦值，然后求两个向量之间的角度

// 特征点反投影,其实得到的是在各自相机坐标系下的一个非归一化的方向向量,和这个点的反投影射线重合
            auto xn1 = pCamera1->unprojectMat_(kp1.pt);
            auto xn2 = pCamera2->unprojectMat_(kp2.pt);
            // 由相机坐标系转到世界坐标系(得到的是那条反投影射线的一个同向向量在世界坐标系下的表示,还是只能够表示方向)，得到视差角余弦值
            auto ray1 = Rwc1*xn1;
            auto ray2 = Rwc2*xn2;


const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur>=0);
bStereo1为有当前基线校准的双目或者有匹配的特征点在右目中的像素坐标

bStereo2为有相邻帧基线校准的双目或者有匹配的特征点在右目中的像素坐标

7.恢复地图点

反投影的向量之间夹角+1生成一个较大的初始值。地图点反投影生成向量

8.检测生成的3D点是否在视野中，设置阈值检查地图点在当前关键帧下的重投影误差。计算地图点在另一个关键帧下的重投影误差

添加地图点，并未地图点添加属性

void LocalMapping::SearchInNeighbors()融合当前关键帧和其共视帧的地图点
 1.获取当期关键帧在公示图中邻接关键帧数量大于nn的帧，其中与当前关键帧相连的称为一级关键帧，与一级关键帧相邻的称为二级关键帧。

2.遍历一级关键帧，进行融合

//?这边是不是有bug 不是应该是!pKFI->isBad&&pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId吗？

if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        // 加入一级相邻关键帧    
        vpTargetKFs.push_back(pKFi);
        // 标记已经加入
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
3.遍历二级关键帧，二级关键帧需要和当前帧没有融合并且也不是当前帧。

4.将当前帧和一级关键帧、二级关键帧融合，正向 将地图点匹配关键帧的特征点 ，如果有对应的地图点则选择观测数目多的替换地图点。如果没有对应的地图点则投影，

 matcher.Fuse(pKFi,vpMapPointMatches);
        if(pKFi->NLeft != -1) matcher.Fuse(pKFi,vpMapPointMatches,true);
将一级二级相邻关键帧地图点分别与当前关键帧地图点进行融合，反向

遍历一级关键帧和二级关键帧并存储地图点并找到需要让融合的加入

5.更新地图点的属性并连接其他帧

LoopClosing闭环
这一部分和orb-slam2不太一样 



主函数run 
Loopclosing中的关键帧是LocalMapping发送过来的，在LocalMapping中通过 InsertKeyFrame 将关键帧插入闭环检测队列mlpLoopKeyFrameQueueLocalMapping是Tracking中发过来的

void LoopClosing::Run() {
    while (1) {
        if (CheckNewKeyFrames()) {

// Step 2 检测有没有共视的区域

bool bDetected = NewDetectCommonRegions();

              if(bDetected){
                if (ComputeSim3()) {
                    CorrectLoop();
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

 

 1 查看闭环检测队列mlpLoopKeyFrameQueue中有没有关键帧进来

      if(CheckNewKeyFrames())
        {
            if(mpLastCurrentKF) 
            {
                mpLastCurrentKF->mvpLoopCandKFs.clear();
                mpLastCurrentKF->mvpMergeCandKFs.clear();
            }
2.检测有没有共视区域//检测有没有回环  就是orb-slam里的deteDetectLoop

3.1如果有检测到融合

imu模式但是没有初始化则放弃合并地图

imu没有初始化就放弃融合
                    if ((mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO) &&
                        (!mpCurrentKF->GetMap()->isImuInitialized()))
                    {
                        cout << "IMU is not initilized, merge is aborted" << endl;
                    }
得到融合的地图位姿，

如果imu模式且初始化成功或者非imu格式

如果是imu模式但是

if(mSold_new.scale()<0.90||mSold_new.scale()>1.1){
2.则认为误差太大放弃融合。如果imu模式下进行过初始化则强制将焊接变换的 roll 和 pitch 设为0

//这里不太懂

imu模式选择MergeLocal2();纯视觉模式选择MergeLocal();

3.检测到回环，但没检测到融合则矫正

3.1如果是imu模式

        if(mpCurrentKF->GetMap()->IsInertial())
// 拿到当前关键帧相对于世界坐标系的位姿
                        cv::Mat Twc = mpCurrentKF->GetPoseInverse();
                        g2o::Sim3 g2oTwc(Converter::toMatrix3d(Twc.rowRange(0, 3).colRange(0, 3)),Converter::toVector3d(Twc.rowRange(0, 3).col(3)),1.0);
这里看不太懂

 // 这里算是通过imu重力方向验证回环结果, 如果pitch或roll角度偏差稍微有一点大,则回环失败. 对yaw容忍比较大(20度)
                        if (fabs(phi(0))<0.008f && fabs(phi(1))<0.008f && fabs(phi(2))<0.349f)
                        {
                            // 如果是imu模式
                            if(mpCurrentKF->GetMap()->IsInertial())
                            {
                                // If inertial, force only yaw
                             
                                // 如果是imu模式,强制将焊接变换的的 roll 和 pitch 设为0
                                if ((mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO) &&
                                        mpCurrentKF->GetMap()->GetIniertialBA2()) // TODO, maybe with GetIniertialBA1
                                {
                                    phi(0)=0;
                                    phi(1)=0;
                                    g2oSww_new = g2o::Sim3(ExpSO3(phi),g2oSww_new.translation(),1.0);
                                    mg2oLoopScw = g2oTwc.inverse()*g2oSww_new;
                                }
                            }
开始回环

纯视觉模式直接开始回环

// 查看是否有外部线程请求复位当前线程
        ResetIfRequested();
 
        // 查看外部线程是否有终止当前线程的请求,如果有的话就跳出这个线程的主函数的主循环
        if(CheckFinish()){
            // cout << "LC: Finish requested" << endl;
            break;
        }
 
        usleep(5000);
    }
bool LoopClosing::NewDetectCommonRegions()闭环检测
// Step 1 从队列中取出一个关键帧,作为当前检测共同区域的关键帧
        unique_lock<mutex> lock(mMutexLoopQueue);
        // 从队列头开始取，也就是先取早进来的关键帧
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        // 取出关键帧后从队列里弹出该关键帧
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        // 设置当前关键帧不要在优化的过程中被删除
        mpCurrentKF->SetNotErase();
        mpCurrentKF->mbCurrentPlaceRecognition = true;
        // 当前关键帧对应的地图
        mpLastMap = mpCurrentKF->GetMap();
    }
imu模式下没有getiniertial则不进行区域检测、双目模式下当前地图帧数少于5不考虑、当前地图关键帧少于12不进行检测

初始化时mnLoopNumCoincidences=0

 bLoopDetectedInKF = false;
 // 递增失败的时序验证次数
            mnLoopNumNotFound++;
       //若果连续两帧时序验证失败则整个回环检测失败
            if(mnLoopNumNotFound >= 2)
初始帧之后

  if(mnLoopNumCoincidences > 0)
得到位姿
通过把候选帧局部窗口内的地图点向新进来的关键帧投影来验证回环检测结果

mnLoopNumCoincidences是成功几何验证的帧数，超过3就认为最终验证成功（mbLoopDetected=true），不超过继续进行时序验证

bool LoopClosing::DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                 std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
 
    set<MapPoint*> spAlreadyMatchedMPs;
    // 把候选帧局部窗口内的地图点投向新进来的当前关键帧,看是否有足够的匹配点
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
 
 
    int nProjMatches = 30;
    int nProjOptMatches = 50;
    int nProjMatchesRep = 100;
// 如果大于一定的数量
    if(nNumProjMatches >= nProjMatches)
    {
        // 为OptimizeSim3接口准备数据
        cv::Mat mScw = Converter::toCvMat(gScw);
        cv::Mat mTwm = pMatchedKF->GetPoseInverse();
        g2o::Sim3 gSwm(Converter::toMatrix3d(mTwm.rowRange(0, 3).colRange(0, 3)),Converter::toVector3d(mTwm.rowRange(0, 3).col(3)),1.0);
        g2o::Sim3 gScm = gScw * gSwm;
        Eigen::Matrix<double, 7, 7> mHessian7x7;
 
 
        // 单目情况下不锁定尺度
        bool bFixedScale = mbFixScale;       // TODO CHECK; Solo para el monocular inertial
        // 如果是imu模式且未完成初始化,不锁定尺度
        if(mpTracker->mSensor==System::IMU_MONOCULAR && !pCurrentKF->GetMap()->GetIniertialBA2())
            bFixedScale=false;
        // 继续优化 Sim3
        int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pMatchedKF, vpMatchedMPs, gScm, 10, bFixedScale, mHessian7x7, true);
 
        // 若匹配的数量大于一定的数目
        if(numOptMatches > nProjOptMatches)
        {
            //!bug, 以下gScw_estimation应该通过上述sim3优化后的位姿来更新。以下mScw应该改为 gscm * gswm.t
            g2o::Sim3 gScw_estimation(Converter::toMatrix3d(mScw.rowRange(0, 3).colRange(0, 3)),
                           Converter::toVector3d(mScw.rowRange(0, 3).col(3)),1.0);
 
            vector<MapPoint*> vpMatchedMP;
            vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
            // 再次通过优化后的Sim3搜索匹配点
            nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw_estimation, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
            //cout << "REFFINE-SIM3: Projection with optimize Sim3 from last KF with " << nNumProjMatches << " matches" << endl;
            // 若果大于期望数目,接受这个结果
            if(nNumProjMatches >= nProjMatchesRep)
            {
                gScw = gScw_estimation;
                // 验证成功
                return true;
            }
        }
    }
    // 验证失败
    return false;
所以共同区域检测时把局部窗口内的地图点向新进来的关键帧投影来验证回环检测结果并优化sim位姿

 // 如果找到共同区域(时序验证成功一次)
        if(bCommonRegion)
        {
            //标记时序检验成功一次 
            bLoopDetectedInKF = true;
            // 累计正检验的成功次数
            mnLoopNumCoincidences++;
            // 不再参与新的回环检测
            mpLoopLastCurrentKF->SetErase();
            mpLoopLastCurrentKF = mpCurrentKF;
            mg2oLoopSlw = gScw; // 记录当前优化的结果为{last T_cw}即为 T_lw
            // 记录匹配到的点
            mvpLoopMatchedMPs = vpMatchedMPs;
 
            // 如果验证数大于等于3则为成功回环
            mbLoopDetected = mnLoopNumCoincidences >= 3;
            // 记录失败的时序校验数为0
            mnLoopNumNotFound = 0;
            //! 这里的条件反了,不过对功能没什么影响,只是打印信息
            if(!mbLoopDetected)
            {
                //f_succes_pr << mpCurrentKF->mNameFile << " " << "8"<< endl;
                //f_succes_pr << "% Number of spatial consensous: " << std::to_string(mnLoopNumCoincidences) << endl;
                cout << "PR: Loop detected with Reffine Sim3" << endl;
            }
若校验成功则把当前帧添加进数据库,且返回true表示找到共同区域
 
    if(mbMergeDetected || mbLoopDetected)
    {
        //f_time_pr << "Geo" << " " << timeGeoKF_ms.count() << endl;
        mpKeyFrameDB->add(mpCurrentKF);
        return true;
    }
若当前关键帧没有被检测到回环或融合,则分别通过bow拿到当前帧最好的三个回环候选帧和融合候选帧

    vector<KeyFrame*> vpMergeBowCand, vpLoopBowCand;
 if(!bMergeDetectedInKF || !bLoopDetectedInKF)
mpKeyFrameDB->DetectNBestCandidates(mpCurrentKF, vpLoopBowCand, vpMergeBowCand,3);
 如果检测到当前帧存在回环或者融合，则把关键帧加入到关键帧数据库中，如果没有检测到关键帧的回环或者融合则抛弃关键帧

bool LoopClosing::DetectAndReffineSim3FromLastKF
把候选帧局部窗口内的地图点投向新进来的当前关键帧,看是否有足够的匹配点，如果大于一定的数量。为OptimizeSim3接口准备数据，计算sim3优化。

bool LoopClosing::DetectCommonRegionsFromLastKF 用来验证候选帧的函数

int LoopClosing::FindMatchesByProjection将窗口内的特征线投到关键帧上寻找匹配点。

FindMatchesByProjection函数会提取候选匹配帧的共视帧集合，以及二级共视帧集合，作为一个大map，在当前帧中寻找2d匹配点，返回匹配点数量。
从候选帧中找五个共视关系最好的点push_bback组成一个局部窗口，拿出当前帧的共视关键帧

//后面的代码咋和别人不一样呢

void LoopClosing::CorrectLoop()回环矫正
 1.停止全局ba。根据共视关系更新与其他关键帧的链接。

2.当前关键帧和与其共视的关键帧形成关键帧组

    // CorrectedSim3：存放闭环g2o优化后当前关键帧的共视关键帧的世界坐标系下Sim3 变换
    // NonCorrectedSim3：存放没有矫正的当前关键帧的共视关键帧的世界坐标系下Sim3 变换
    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
3：通过mg2oLoopScw（认为是准的）来进行位姿传播，得到当前关键帧的共视关键帧的世界坐标系下Sim3 位姿（还没有修正） 得到矫正的当前关键帧的共视关键帧位姿后，修正这些关键帧的地图点。

4.将共视关键帧的Sim3转换为SE3，根据更新的Sim3，更新关键帧的位姿。检查当前帧的地图点与经过闭环匹配后该帧的地图点是否存在冲突，对冲突的进行替换或填补。

5.将闭环相连关键帧组mvpLoopMapPoints 投影到当前关键帧组中，进行匹配，融合，新增或替换当前关键帧组中KF的地图点

6.更新当前关键帧之间的共视相连关系，得到因闭环时MapPoints融合而新得到的连接关系

遍历当前帧相连关键帧组（一级相连）。得到与当前帧相连关键帧的相连关键帧（二级相连）更新一级相连关键帧的连接关系(会把当前关键帧添加进去,因为地图点已经更新和替换了)。取出该帧更新后的连接关系。从连接关系中去除闭环之前的二级连接关系，剩下的连接就是由闭环得到的连接关系。从连接关系中去除闭环之前的一级连接关系，剩下的连接就是由闭环得到的连接关系

7.添加当前帧与闭环匹配帧之间的边（这个连接关系不优化）

8.新建一个线程用于全局BA优化，只有进行全局ba

void LoopClosing::MergeLocal2()惯性模式下的地图融合
1.如果正在进行全局ba，暂停 

设置指针指向当前关键帧的地图和需要融合的关键帧地图的指针

    // 当前关键帧地图的指针
    Map* pCurrentMap = mpCurrentKF->GetMap();
    // 融合关键帧地图的指针
    Map* pMergeMap = mpMergeMatchedKF->GetMap();
    // Step 3 利用前面计算的坐标系变换位姿，把整个当前地图（关键帧及地图点）变换到融合帧所在地图
    {
        // 把当前关键帧所在的地图位姿带到融合关键帧所在的地图
        // mSold_new = gSw2w1 记录的是当前关键帧世界坐标系到融合关键帧世界坐标系的变换
        float s_on = mSold_new.scale();
        cv::Mat R_on = Converter::toCvMat(mSold_new.rotation().toRotationMatrix());
        cv::Mat t_on = Converter::toCvMat(mSold_new.translation());
        // 锁住altas更新地图
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        // 队列里还没来得及处理的关键帧清空
        mpLocalMapper->EmptyQueue();
 
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        // 是否将尺度更新到速度
        bool bScaleVel=false;
        if(s_on!=1) //?判断浮点数和1严格相等是不是不合适？
            bScaleVel=true;
        // 利用mSold_new位姿把整个当前地图中的关键帧和地图点变换到融合帧所在地图的坐标系下
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(R_on,s_on,bScaleVel,t_on);
        // 尺度更新到普通帧位姿
        mpTracker->UpdateFrameIMU(s_on,mpCurrentKF->GetImuBias(),mpTracker->GetLastKeyFrame());
 
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    }
4。如果imu没有完成初始化则进行imu快速优化

5.融合帧所在地图里的关键帧和地图点删除变为当前关键帧所在地图

6.融合新旧地图的生成树

7 把融合关键帧的共视窗口里的地图点投到当前关键帧的共视窗口里，把重复的点融合掉

8 针对缝合区域的窗口内进行进行welding BA
