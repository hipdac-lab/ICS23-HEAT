<h3 align="center"><img src="https://user-images.githubusercontent.com/5705572/231052807-a41e7543-28f6-4405-b4de-8ee9b136c6be.gif" width="150"></h3>

<h3 align="center">
A High-Performance Training System for Collaborative Filtering Based Recommendation on CPUs
</h3>

HEAT is a <ins>H</ins>ighly <ins>E</ins>fficient and <ins>A</ins>ffordable <ins>T</ins>raining system designed for collaborative filtering-based recommendations on multi-core CPUs, utilizing the SimpleX approach [1]. The system incorporates three main optimizations: (1) Tiling the embedding matrix to enhance data locality and minimize cache misses, thereby reducing read latency; (2) Streamlining stochastic gradient descent (SGD) with sampling by parallelizing vector products, specifically the similarity computation, instead of matrix-matrix multiplications, in order to eliminate memory copies for matrix data preparation; and (3) Aggressively reusing intermediate results from the forward phase during the backward phase to mitigate redundant computation. For more information, please refer to our [technical paper](https://github.com/hipdac-lab/HEAT/blob/main/ICS23-HEAT.pdf) [2].

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
Change ```data_dir``` in the below three config0.yaml files to their corresponding data paths, e.g., ```/home/dingwen.tao/HEAT/LightGCN/Data/amazon-book```, ```/home/dingwen.tao/HEAT/LightGCN/Data/yelp2018```, and ```/home/dingwen.tao/HEAT/LightGCN/Data/gowalla```, respectively.

Finally, run the tests:
```
    cd cf_cpu/cf
    python main.py --config ./benchmarks/AmazonBooks/MF_CCL/configs/config0.yaml
    python main.py --config ./benchmarks/Yelp18/MF_CCL/configs/config0.yaml
    python main.py --config ./benchmarks/Gowalla/MF_CCL/configs/config0.yaml
```

### Step 6: Exptected output
```
this is main ...
{'num_negs': 16, 'max_his': 100, 'embedding_dim': 64, 'neg_sampler': 0, 'tile_size': 512, 'refresh_interval': 8192, 'embedding_dropout': 0, 'similarity_score': 'dot', 'loss': 'PairwiseLogisticLoss', 'metrics': ['Recall(k=20)', 'Recall(k=50)', 'NDCG(k=20)', 'NDCG(k=50)', 'HitRate(k=20)', 'HitRate(k=50)'], 'optimizer': 'sgd', 'milestones': [10], 'learning_rate': 0.01, 'clip_val': 1.0, 'embedding_regularizer': 1e-07, 'net_regularizer': 0, 'net_dropout': 0, 'epochs': 5, 'eval_interval': 2, 'seed': 2022}
gen dataset info of /home/dingwen.tao/HEAT/LightGCN/Data/amazon-book/train.txt 
number of users: 52643; min_user_id: 0; max_user_id: 52642
number of items: 91599; min_item_id: 0; max_item_id: 91598
total samples: 2380730 
update config, init c_instance !!! 


Warning 13647 has 0 items !!! 
Warning 41589 has 0 items !!! 
Warning 50736 has 0 items !!! 
Warning 52234 has 0 items !!! 
gen dataset info of /home/dingwen.tao/HEAT/LightGCN/Data/amazon-book/test.txt 
Warning item_id is not continuous! 
number of users: 52643; min_user_id: 0; max_user_id: 52642
number of items: 82629; min_item_id: 0; max_item_id: 91598
total samples: 603378 


iterations 2380730 max_threads 8
epoch time: 244.985 s 
epoch: 0; loss: 1.35187828540802; epoch_time: 244.98503732681274
iterations 2380730 max_threads 8
epoch time: 244.695 s 
epoch: 1; loss: 0.6897922158241272; epoch_time: 244.69532942771912
iterations 2380730 max_threads 8
epoch time: 246.372 s 
epoch: 2; loss: 0.5268339514732361; epoch_time: 246.37257051467896
--- Start evaluation ---
sim_matrix shape: (52643, 91599) !!! 
iterations 2380730 max_threads 8
epoch time: 246.134 s 
epoch: 3; loss: 0.45124438405036926; epoch_time: 246.13380026817322
iterations 2380730 max_threads 8
epoch time: 245.861 s 
epoch: 4; loss: 0.40876930952072144; epoch_time: 245.86116886138916
...
```

## References
- [1] Kelong Mao, Jieming Zhu, Jinpeng Wang, Quanyu Dai, Zhenhua Dong, Xi Xiao, and Xiuqiang He. "SimpleX: A simple and strong baseline for collaborative filtering." In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pp. 1243-1252. 2021.
- [2] Chengming Zhang, Shaden Smith, Baixi Sun, Jiannan Tian, Jonathan Soifer, Xiaodong Yu, Shuaiwen Leon Song, Yuxiong He, Dingwen Tao. "HEAT: A Highly Efficient and Affordable Training System for Collaborative Filtering Based Recommendation on Multi-core CPUs." In Proceedings of the 37th ACM International Conference on Supercomputing, pp. xx-xx. 2023.
