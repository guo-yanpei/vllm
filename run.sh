export CMAKE_CUDA_ARCHITECTURES=90
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_VISIBLE_DEVICES=2

python setup.py install
echo $1
cp $1 build/lib.linux-x86_64-cpython-310/
cd build/lib.linux-x86_64-cpython-310 && python $1