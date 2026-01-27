# motion_controller
Motion controller for ROBOTIS Physical AI Lineup

- Install OSQP
```bash
git clone https://github.com/osqp/osqp
cd osqp
mkdir build && cd build
cmake -G "Unix Makefiles" ..
cmake --build .
sudo cmake --build . --target install
```
-  Install osqp-eigen
```bash
git clone https://github.com/robotology/osqp-eigen.git
cd osqp-eigen
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=<custom-folder> ../
make
sudo make install
```
- Set library path
```bash
echo "export OsqpEigen_DIR=/path/to/<custom-folder>" >> ~/.bashrc && \
echo "export LD_LIBRARY_PATH=/path/to/<custom-folder>/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
```
- Install Pinocchio
```bash
sudo apt update
sudo apt install ros-jazzy-pinocchio
```