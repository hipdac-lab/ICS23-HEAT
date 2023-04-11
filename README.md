# HEAT user guide

## Code structure
```
    The C++ backend is in cf_cpu/src. The Python frontend is in the cf_cpu/cf.
    We use CMake to build C++ backend into .so shared library, and import the library into Python frontend. 
```

## Build Instructions

### Step 1: Install dependencies
Install Python3, CMake
```
    conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
    pip install scikit-learn==1.0
    pip install pandas==1.3.4
    pip install PyYAML==5.4.1
    pip install h5py==3.5.0
    pip install tqdm==4.62.2
```

### Step 2: Clone HEAT and build pybind and eigen dependencies
Please first set compiler, by export CC and CXX.
```
    git clone https://github.com/hipdac-lab/HEAT.git
    cd HEAT
    git submodule update --init --recursive
    cd cf_cpu/extern/pybind11
    mkdir build
    cd build
    cmake ..
    make check -j
    cd ../../eigen/
    mkdir build
    cd build/
    cmake ..
```

### Step 3: Build HEAT
```
    cd ../../../ #go to HEAT/cf_cpu folder
    mkdir build
    cd build
    cmake ..
    make -j
    cp *.so ../cf/
```

### Step 4: Download sample datasets
```
    cd ../../ #go to HEAT root folder
    git clone https://github.com/kuandeng/LightGCN.git
```

### Step 5: Run HEAT
Change ```data_dir``` in the below three config0.yaml files to their corresponding data paths, e.g., ```HEAT/LightGCN/Data/amazon-book```, ```HEAT/LightGCN/Data/yelp2018```, and ```HEAT/LightGCN/Data/gowalla```, respectively.

Finally, run the tests:
```
    cd cf_cpu/cf
    python main.py --config ./benchmarks/AmazonBooks/MF_CCL/configs/config0.yaml
    python main.py --config ./benchmarks/Yelp18/MF_CCL/configs/config0.yaml
    python main.py --config ./benchmarks/Gowalla/MF_CCL/configs/config0.yaml
```
