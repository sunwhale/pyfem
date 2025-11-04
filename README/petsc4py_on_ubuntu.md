# petsc4py安装
参考：[https://petsc.org/release/install/](https://petsc.org/release/install/)

## 1. 安装miniconda
```shell
sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
sudo sed -i '$aexport PATH=/opt/miniconda3/bin:$PATH' /etc/profile # 写入环境变量
source /etc/profile
conda init
exit
```

## 2. 建立虚拟环境pyfem
```shell
conda create -n pyfem -y python==3.11
conda activate pyfem
git clone https://gitee.com/sunwhale/pyfem.git
cd pyfem
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ # -i https://pypi.org/simple # 官方源
python -c "import numpy; print(numpy.get_include())" # 查看numpy包头文件路径
```

## 3. 安装petsc
```shell
sudo apt install -y git gcc g++ gfortran make
git clone -b release https://gitlab.com/petsc/petsc.git petsc
cd petsc
./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich --download-fblaslapack
make all check
sudo sed -i '$aexport PETSC_DIR=/home/dell/petsc' ~/.bashrc # 写入环境变量
sudo sed -i '$aexport PETSC_ARCH=arch-linux-c-debug' ~/.bashrc # 写入环境变量
sudo sed -i '$aexport NUMPY_INCLUDE=/home/dell/.conda/envs/pyfem312/lib/python3.12/site-packages/numpy/_core/include' ~/.bashrc # 写入环境变量
sudo sed -i '$aexport PATH=/home/dell/petsc/arch-linux-c-debug/bin:$PATH' ~/.bashrc # 写入环境变量
source ~/.bashrc
exit
```

## 4. 安装petsc4py
```shell
conda activate pyfem
cd petsc
python -m pip install src/binding/petsc4py
```

## 5. 安装mpich4py
```shell
env MPICC=/home/sunjingyu/petsc/arch-linux-c-debug/bin/mpicc python -m pip install mpi4py
```
或者
```shell
export MPICC=/home/dell/petsc/arch-linux-c-debug/bin/mpicc
pip install --no-binary=mpi4py mpi4py
```