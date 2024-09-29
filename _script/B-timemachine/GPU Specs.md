# Paddle Installation Problem
## Check GPU CUDA Version
``` nvcc --version```
```module list``` To see current loaded cuda version.
```module avail cuda```  To check for available cuda version 
```module load cuda/11.8```
Reinstall Numpy: `pip install --upgrade --force-reinstall numpy==1.24.3`

## ERROR NOTES
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
paddlepaddle-gpu 3.0.0b1 requires nvidia-cublas-cu11==11.11.3.6; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
paddlepaddle-gpu 3.0.0b1 requires nvidia-cuda-cupti-cu11==11.8.87; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
paddlepaddle-gpu 3.0.0b1 requires nvidia-cuda-nvrtc-cu11==11.8.89; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
paddlepaddle-gpu 3.0.0b1 requires nvidia-cuda-runtime-cu11==11.8.89; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
paddlepaddle-gpu 3.0.0b1 requires nvidia-cudnn-cu11==8.7.0.84; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
paddlepaddle-gpu 3.0.0b1 requires nvidia-cufft-cu11==10.9.0.58; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
paddlepaddle-gpu 3.0.0b1 requires nvidia-curand-cu11==10.3.0.86; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
paddlepaddle-gpu 3.0.0b1 requires nvidia-cusolver-cu11==11.4.1.48; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
paddlepaddle-gpu 3.0.0b1 requires nvidia-cusparse-cu11==11.7.5.86; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
paddlepaddle-gpu 3.0.0b1 requires nvidia-nccl-cu11==2.19.3; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.