# Detectron2 Tutorials
This is repository for basic tutorials to configure detectron2 i.e. an open source library of object detection by Facebook [Detectron2](https://github.com/facebookresearch/detectron2)

## Installation Guide

1. Install `Python ≥ 3.6` 
   ```
   $ sudo apt-get update
   $ sudo apt-get install python3.6
   ```
2. OS Should be **Linux** or **macOS**
3. Installation CUDA (≥10) - _This tutorial is for CUDA-11.4_
   - Go to [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads).
   - Select Compatible Variables `(Linux > x86_64 > Ubuntu > 18.04 > deb[local])`
   - Download Required Version of CUDA
      ```
      wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-ubuntu1804-11-4-local_11.4.0-470.42.01-1_amd64.deb
      ```
   - Install Downloaded Packge 
     ```
     sudo dpkg -i cuda-repo-ubuntu1804-11-4-local_11.4.0-470.42.01-1_amd64.deb
     ```
   - Adding Key 
     ```
     sudo apt-key add /var/cuda-repo-ubuntu1804-11-4-local/7fa2af80.pub
     ```
   - Installation of CUDA-11.4
     ```
     sudo apt-get update
     sudo apt install cuda-11.4
     ```
4. Setting CUDA Environment
   - Validate CUDA Installation 
      ```
      cd /usr/local
      ls
      ```
      You will find installed CUDA in listed files
   - Go to your user directory 
      ```
      cd /home/{USER}
      ```
      Replace `{USER}` with your user name
   - Add Environment Variables in `.bashrc` file in `/home/{USER}`
      ```
      nano /home/{USER}/.bashrc
      ```
      Editor will be opened in your command line
   - Add these lines in file
      ```
      export PATH="/usr/local/cuda-11.4/bin:$PATH"
      export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
      ```
      `Ctrl + X` > `Y` > `Enter`
   - Add variables in `/etc/environment`
      ```
      sudo nano/etc/environment
      ```
   - Add Following lines 
     If variables already there, so add this part at the end of line of `PATH` inside the `" "` inverted commas
     ```
     :/usr/local/cuda-11.4/bin
     ```
     If nothing there, just paste whole line there
     ```
     PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-11.4/bin"
     ```
   - Run to check CUDA Version 
     ```
     nvcc --version
     ```
     If still faceing any problem, Log out and log in again
5. Create Virtual Environment
   - Create Environment 
     ```
     virtualenv -p pytho3 {ENV-NAME}
     ```
     Replace `{ENV-NAME}` with Environment Name you want
   - Activate the environment 
     ```
     {ENV-NAME}/bin/activate
     ```
     Now you will have your `{ENV-NAME}` in begining of your command line
6. Download **Detectron2**
   Clone Detectron2 repository 
   ```
   git clone https://github.com/facebookresearch/detectron2.git
   cd detectron2
   ```
7. Downloading Pytorch compatible with your CUDA Version, as we downloaded CUDA-11.4, so we download Pytorch with CUDA-11.4.
   - Go to [Pytorch](https://pytorch.org/)
   - Setect your variables `Stable(1.9.0) > Linux > Pip > Python > CUDA 11.1`
   - You will find command
     ```
     pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
     ```
     Now you have same version of CUDA and Pytorch compatible CUDA Version
   - To validate previous condition, Run following command 
     ```
     python -m detectron2.utils.collect_env
     ```
     You can Check CUDA compiler & Pytorch built CUDA Runtime same
8. Install Required Libraries and Packages 
   - OpenCV
     ```
     pip install opencv-python 
     ```
   - Cython
     ```
     pip install cython
     ```
   - COCO Python API
     ```
     pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
     ```
9. Check GCC Version 
   ```
   gcc --version
   ```
   should be greater than 7.
10. Build Detectron2
    ```
    python -m pip install -e detectron2
    ```
    Will cause problem if you don't have same version of Pytorch with your CUDA
    You can validate succesfull build by finding "build" folder in directory.
    (For Linux Users) You can find pre-build Detectron2 from [Pre Build Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only)
    
11. Now you can Get Started with Detectron :-). Further Documentation [Detectron2 Documentation](https://detectron2.readthedocs.io/en/latest/index.html)




