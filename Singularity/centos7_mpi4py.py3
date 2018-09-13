BootStrap: docker
From: centos:latest

%post
yum -y update
 yum -y groupinstall 'Development Tools'
 yum -y install bind-utils
 yum -y install infiniband-diags
 yum -y install epel-release
 yum -y install vim
 yum -y install net-tools
 yum -y install nmap-ncat
 yum -y install telnet
 yum -y install wget tar
 yum -y install mlocate 
 yum -y install curl file git 
 yum -y install strace

# installing miniconda + python 3
wget https://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p /usr/local/miniconda

# installing git
yum -y install git-core

 # Require python 2.7
 echo "Python version: "
 python -V

 yum -y install python-devel
 yum -y install python-pip
 yum -y install yum-utils
 yum -y install tkinter
 yum -y install lapack-devel lapack blas blas-devel

# Install recent gcc 
 yum -y install centos-release-scl
 yum -y install devtoolset-7-gcc*
 scl enable devtoolset-7 bash
 echo "gcc version: ************"
 which gcc
 gcc --version

 yum -y install cmake
 
# OpenMPI v.3.x compatibilty
 yum -y install m4 autoconf automake libtool flex

# Python pip installations (MITIM compatibility)
 pip install --upgrade pip

# ----------- OpenMPI installation ------------
 yum -y install libxml2 libxml2-devel libxslt libxslt-devel

 echo "Installing OpenMPI into container..."

 # Here we are at the base, /, of the container
 rm -rf openmpi-3*
 wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.1.tar.gz 
 tar zxvf openmpi-3.1.1.tar.gz

 # Now at /ompi
 cd openmpi-3.1.1
 ./configure --prefix=/usr/local  --enable-install-libpmix  --with-pmix=internal
 make
 make install

 echo "###### Now installing mpi4py ########"
 MPICC=/usr/local/bin/mpicc pip install --no-cache-dir mpi4py


 #installing for python 3 as well 
 
 mkdir /screen
 mkdir /pasteur
 mkdir -p /local/scratch

%environment
PATH=/usr/local/openmpi/bin:/usr/local/miniconda/bin:/usr/local/bin:$PATH
export PATH
LD_LIBRARY_PATH=/usr/local/openmpi/lib:/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH


