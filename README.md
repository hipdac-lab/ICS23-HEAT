<h3 align="center"><img src="https://user-images.githubusercontent.com/5705572/231052807-a41e7543-28f6-4405-b4de-8ee9b136c6be.gif" width="150"></h3>

<h3 align="center">
A High-Performance Training System for Collaborative Filtering Based Recommendation on CPUs
</h3>

HEAT is a <ins>H</ins>ighly <ins>E</ins>fficient and <ins>A</ins>ffordable <ins>T</ins>raining system designed for collaborative filtering-based recommendations on multi-core CPUs, utilizing the SimpleX approach [1]. The system incorporates three main optimizations: (1) Tiling the embedding matrix to enhance data locality and minimize cache misses, thereby reducing read latency; (2) Streamlining stochastic gradient descent (SGD) with sampling by parallelizing vector products, specifically the similarity computation, instead of matrix-matrix multiplications, in order to eliminate memory copies for matrix data preparation; and (3) Aggressively reusing intermediate results from the forward phase during the backward phase to mitigate redundant computation. For more information, please refer to our [technical paper](https://eecs.wsu.edu/~dtao/paper/ICS23-HEAT.pdf) [2].

## Code Structure
```
    HEAT's C++ backend is in cf_cpu/src, while its Python frontend is in the cf_cpu/cf. 
    We use CMake to build C++ backend into .so shared library, and import the library into Python frontend. 
```

## Build and Test

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

## References
- [1] Kelong Mao, Jieming Zhu, Jinpeng Wang, Quanyu Dai, Zhenhua Dong, Xi Xiao, and Xiuqiang He. "SimpleX: A simple and strong baseline for collaborative filtering." In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pp. 1243-1252. 2021.
- [2] Chengming Zhang, Shaden Smith, Baixi Sun, Jiannan Tian, Jonathan Soifer, Xiaodong Yu, Shuaiwen Leon Song, Yuxiong He, Dingwen Tao. "HEAT: A Highly Efficient and Affordable Training System for Collaborative Filtering Based Recommendation on Multi-core CPUs." In Proceedings of the 37th ACM International Conference on Supercomputing, pp. xx-xx. 2023.
