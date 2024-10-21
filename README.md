# Image_Recognition_Project
图像识别智能算法

# script
script中存储脚本，对图片进行编辑
spin.py用来旋转正方形和三角形，其边缘进行复制扩展

# code
code中存储训练代码，原始代码

# game_code
gamecode中存储比赛段代码

# static
代码测试时前端存储文件

# templates
前端模板

# app.py
端口程序

#遇到问题

## 问题1：上传到podman不能导入镜像（镜像过大，服务器内存小）
解决方案：
    1、不改变环境，将原环境没有用到的依赖全删掉
    2、使用新的基本依赖，使用小版本python镜像，下载包时不下载推荐包
    
    使用的另外一个方法，下载官方的tensorflow镜像包，tf=2.2.0,py=3.6,拉取命令docker pull tensorflow/tensorflow:2.2.0

    ### tensorflow官方镜像包
    Package                Version
    ---------------------- -----------
    absl-py                0.9.0
    asn1crypto             0.24.0
    astunparse             1.6.3
    cachetools             4.1.0
    certifi                2020.4.5.1
    chardet                3.0.4
    cryptography           2.1.4
    gast                   0.3.3
    google-auth            1.14.2
    google-auth-oauthlib   0.4.1
    google-pasta           0.2.0
    grpcio                 1.28.1
    h5py                   2.10.0
    idna                   2.6
    Keras-Preprocessing    1.1.0
    keyring                10.6.0
    keyrings.alt           3.0
    Markdown               3.2.1
    numpy                  1.18.4
    oauthlib               3.1.0
    opt-einsum             3.2.1
    pip                    20.1
    protobuf               3.11.3
    pyasn1                 0.4.8
    pyasn1-modules         0.2.8
    pycrypto               2.6.1
    pygobject              3.26.1
    pyxdg                  0.25
    requests               2.23.0
    requests-oauthlib      1.3.0
    rsa                    4.0
    scipy                  1.4.1
    SecretStorage          2.3.1
    setuptools             46.1.3
    six                    1.14.0
    tensorboard            2.2.1
    tensorboard-plugin-wit 1.6.0.post3
    tensorflow             2.2.0
    tensorflow-estimator   2.2.0
    termcolor              1.1.0
    urllib3                1.25.9
    Werkzeug               1.0.1
    wheel                  0.30.0
    wrapt                  1.12.1

## 问题2：服务器不支持avx虚拟指令
解决方案：
    回退早期tf版本