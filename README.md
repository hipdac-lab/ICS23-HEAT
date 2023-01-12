# HEAT user guide

### Step 1: Install dependencies
1. Install Python3, CMake
```
    conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
    pip install scikit-learn==1.0
    pip install pandas==1.3.4
    pip install PyYAML==5.4.1
    pip install h5py==3.5.0
    pip install tqdm==4.62.2
```

### Step 2: Build HEAT
```
    mkdir build
    cd build
    cmake ..
    make -j
```

### Step 3: Run HEAT
```
    python main.py --config ./benchmarks/AmazonBooks/MF_CCL/configs/config0.yaml
    python main.py --config ./benchmarks/Yelp18/MF_CCL/configs/config0.yaml
    python main.py --config ./benchmarks/Gowalla/MF_CCL/configs/config0.yaml
```
