# Sygnal 
这个包是用来在具有噪声的信道中提取信号。

**basic** 包含信号的矩的计算。默认信号和噪声无关联。

**gaussian_noise** 包含在 **高斯噪声假设**下的信号的矩、信号的关联函数计算。

**correlate** 包含1D或2D数组的，特定维度上的卷积计算（full, valid, same, circulate）方式

# Installation
## 1. 安装numpy
```bash
pip install numpy
```
## 2. 安装jax
jax是谷歌开发的支持CPU/CUDA/TPU和JIT技术的张量操作包。
```bash
# CPU版本
pip install --upgrade "jax[cpu]"

# GPU版本
pip install --upgrade pip

# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
需要注意CUDA版本必须要匹配。


## 3. 安装本包
```bash
python setup.py install
```