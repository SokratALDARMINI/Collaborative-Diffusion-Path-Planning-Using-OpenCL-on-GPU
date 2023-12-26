# Collaborative-Diffusion-Path-Planning-Using-OpenCL-on-GPU

## Algorithms

This project represents implementing the path planning algorithm proposed in [1]. The algorithm is called Collaborative Diffusion, designed to be implemented on a GPU using parallel threads for grid maps.

The algorithm is summarized as follows.

![image](https://github.com/SokratALDARMINI/Collaborative-Diffusion-Path-Planning-Using-OpenCL-on-GPU/assets/95107709/bf0454e0-01c7-4d47-9017-176c067ed53f)


The following example shows a map of size 64*64:

![Capture 1](https://github.com/SokratALDARMINI/Collaborative-Diffusion-Path-Planning-Using-OpenCL-on-GPU/assets/95107709/18f0d796-fb7a-40f6-a48a-7be171a641c5)


Result of applying the diffusion algorithm:

![Capture 2](https://github.com/SokratALDARMINI/Collaborative-Diffusion-Path-Planning-Using-OpenCL-on-GPU/assets/95107709/e81d0f06-9405-4f96-8fd8-f03d30b88df3)


Generated path:

![Capture 3](https://github.com/SokratALDARMINI/Collaborative-Diffusion-Path-Planning-Using-OpenCL-on-GPU/assets/95107709/ca72a2a0-e110-4935-890d-7f2dd2d1a949)


Generated path on the original map:

![Capture 4](https://github.com/SokratALDARMINI/Collaborative-Diffusion-Path-Planning-Using-OpenCL-on-GPU/assets/95107709/535af826-d166-49df-a15f-4b1a0bf0919c)


## Instructions:

The kernel program is 'matrixAverageFilter.cl'.

To modify the obstacle generation method, change the value of '#define obstacles_tpye 1' to '0', '1', or '2' in 'main.cpp'.

This project leverages [Boost Compute]([https://myoctocat.com/assets/images/base-octocat.svg](https://github.com/boostorg/compute)) for OpenCL applications. Being exclusively a template/header library, it bypasses the need for additional compilation or linkage. Boost included Compute starting from version 1.61.

### For Linux Users:

Should your system's repositories not have the latest Boost version, execute 'make boost' to download a current version directly into the project.

To build and initiate the program, simply run the 'make' command.

### Windows User Instructions:

AMD Graphics: Obtain the OpenCL SDK via [Github](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases). For Nvidia Graphics: Secure the CUDA Toolkit from [Nvidia’s website](https://developer.nvidia.com/cuda-downloads).

Proceed to download [Boost](http://www.boost.org/) and place it in a newly created subfolder named 'boost', excluding any version numbers.

Given Compute's dependency on libboost_chrono, building all libraries is recommended. In the Visual Studio command prompt, go to the boost folder and run 'bootstrap' and then 'b2' to complete this process.

[1] McMillan, Craig, Emma Hart, and Kevin Chalmers. "Collaborative Diffusion on the GPU for Path-finding in Games." Applications of Evolutionary Computation: 18th European Conference, EvoApplications 2015, Copenhagen, Denmark, April 8-10, 2015, Proceedings 18. Springer International Publishing, 2015.
