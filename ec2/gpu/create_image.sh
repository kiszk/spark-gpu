#!/bin/bash
# Creates an AMI for the Spark EC2 scripts starting with a stock Amazon 
# Linux AMI.
# This has only been tested with Amazon Linux AMI 2014.03.2 

set -e

if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi


# Update
yum update -y
# Dev tools
sudo yum install -y java-1.8.0-openjdk-devel gcc gcc-c++ ant git
# Perf tools
sudo yum install -y dstat iotop strace sysstat htop perf
#sudo debuginfo-install -q -y glibc
#sudo debuginfo-install -q -y kernel
yum install -y kernel-devel-`uname -r`
sudo yum --enablerepo='*-debug*' install -q -y java-1.8.0-openjdk-debuginfo.x86_64

# PySpark and MLlib deps
# sudo yum install -y  python-matplotlib python-tornado scipy libgfortran
sudo yum install -y libgfortran
# SparkR deps
#sudo yum install -y R
# Other handy tools
sudo yum install -y pssh
# Ganglia
#sudo yum install -y ganglia ganglia-web ganglia-gmond ganglia-gmetad

if [ "$1" == "gpu" ]; then
# CUDA
sudo yum install -y gcc-c++
# Install NVIDIA Driver
sudo wget -P /root -q http://us.download.nvidia.com/XFree86/Linux-x86_64/346.96/NVIDIA-Linux-x86_64-346.96.run
sudo chmod +x /root/NVIDIA-Linux-x86_64-346.96.run
sudo /root/NVIDIA-Linux-x86_64-346.96.run -s > /root/driver.log 2>&1
# Install CUDA (without driver installation... for Amazon Linux 2015.09)
sudo wget -P /root -q http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run
sudo chmod +x /root/cuda_7.0.28_linux.run
sudo /root/cuda_7.0.28_linux.run -extract=/root
sudo /root/cuda-linux64-rel-7.0.28-19326674.run -noprompt > /root/cuda.log 2>&1 
rm -f *.run
fi

# Root ssh config
sudo sed -i 's/PermitRootLogin.*/PermitRootLogin without-password/g' \
  /etc/ssh/sshd_config
sudo sed -i 's/disable_root.*/disable_root: 0/g' /etc/cloud/cloud.cfg

# Set up ephemeral mounts
sudo sed -i 's/mounts.*//g' /etc/cloud/cloud.cfg
sudo sed -i 's/.*ephemeral.*//g' /etc/cloud/cloud.cfg
sudo sed -i 's/.*swap.*//g' /etc/cloud/cloud.cfg

echo "mounts:" >> /etc/cloud/cloud.cfg
echo " - [ ephemeral0, /mnt, auto, \"defaults,noatime,nodiratime\", "\
  "\"0\", \"0\" ]" >> /etc/cloud.cloud.cfg

for x in {1..23}; do
  echo " - [ ephemeral$x, /mnt$((x + 1)), auto, "\
    "\"defaults,noatime,nodiratime\", \"0\", \"0\" ]" >> /etc/cloud/cloud.cfg
done

# Install Maven (for Hadoop)
cd /tmp
sudo wget -q "http://archive.apache.org/dist/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz"
sudo tar xf apache-maven-3.3.9-bin.tar.gz
sudo mv apache-maven-3.3.9 /opt

# Edit bash profile
echo "export PS1=\"\\u@\\h \\W]\\$ \"" >> ~/.bash_profile
echo "export JAVA_HOME=/usr/lib/jvm/java-1.8.0" >> ~/.bash_profile
echo "export M2_HOME=/opt/apache-maven-3.3.9" >> ~/.bash_profile
echo "export M2_HOME=/opt/hadoop-2.4.1" >> ~/.bash_profile
echo "export PATH=/usr/local/cuda/bin:\$PATH:\$M2_HOME/bin" >> ~/.bash_profile
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> ~/.bash_profile

source ~/.bash_profile

# Build Hadoop to install native libs
sudo mkdir /root/hadoop-native
cd /tmp
#sudo yum install -y protobuf-compiler cmake openssl-devel
#wget "http://archive.apache.org/dist/hadoop/common/hadoop-2.4.1/hadoop-2.4.1-src.tar.gz"
#tar xvzf hadoop-2.4.1-src.tar.gz
#cd hadoop-2.4.1-src
#mvn package -Pdist,native -DskipTests -Dtar
#sudo mv hadoop-dist/target/hadoop-2.4.1/lib/native/* /root/hadoop-native
sudo wget -q "http://archive.apache.org/dist/hadoop/common/hadoop-2.4.1/hadoop-2.4.1.tar.gz"
sudo tar xf hadoop-2.4.1.tar.gz
sudo mv hadoop-2.4.1 /root/hadoop

# Install Snappy lib (for Hadoop)
yum install -y snappy
sudo ln -sf /usr/lib64/libsnappy.so.1 /root/hadoop-native/.

# Create /usr/bin/realpath which is used by R to find Java installations
# NOTE: /usr/bin/realpath is missing in CentOS AMIs. See
# http://superuser.com/questions/771104/usr-bin-realpath-not-found-in-centos-6-5
sudo echo '#!/bin/bash' > /usr/bin/realpath
sudo echo 'readlink -e "$@"' >> /usr/bin/realpath
sudo chmod a+x /usr/bin/realpath

mkdir -p /tmp/spark-events
chmod 777 /tmp/spark-events

