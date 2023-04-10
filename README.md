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

### Step 2: Clone HEAT and build dependency
```
    git clone https://github.com/hipdac-lab/HEAT.git
    git submodule update --init --recursive
    cd cf_cpu/extern/pybind11
    cd build
    cmake ..
    make check -j 4
```

### Step 3: Build HEAT
Please firt set compiler, CC and CXX. 
```
    cd cf_cpu
    mkdir build
    cd build
    cmake ..
    make -j
    cp .xx.so ../cf/
```

### Step 4: Git datasets
```
    git clone https://github.com/kuandeng/LightGCN.git
```

### Step 5: Run HEAT
Change data path in config file to LightGCN/Data
```
    cd cf_cpu/cf
    python main.py --config ./benchmarks/AmazonBooks/MF_CCL/configs/config0.yaml
    python main.py --config ./benchmarks/Yelp18/MF_CCL/configs/config0.yaml
    python main.py --config ./benchmarks/Gowalla/MF_CCL/configs/config0.yaml
```
