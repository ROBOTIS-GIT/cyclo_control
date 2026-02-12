# motion_controller
Motion controller for ROBOTIS Physical AI Lineup

- Install OSQP
```bash
cd ~/
git clone https://github.com/osqp/osqp
cd ~/osqp
mkdir build && cd build
cmake -G "Unix Makefiles" ..
cmake --build .
sudo cmake --build . --target install
```
-  Install osqp-eigen
```bash
cd ~/
mkdir osqp-eigen_install
git clone https://github.com/robotology/osqp-eigen.git
cd ~/osqp-eigen
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=~/osqp-eigen_install ../
make
sudo make install
```
- Set library path
```bash
echo "export OsqpEigen_DIR=~/osqp-eigen_install" >> ~/.bashrc && \
echo "export LD_LIBRARY_PATH=~/osqp-eigen_install/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
```
- Install Pinocchio
```bash
sudo apt update
sudo apt install ros-jazzy-pinocchio
```