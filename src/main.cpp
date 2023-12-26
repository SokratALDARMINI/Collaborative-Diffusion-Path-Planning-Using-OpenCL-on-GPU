
#include <cmath>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>


#pragma warning(push)
// Disable warning for VS 2017
#pragma warning(disable : 4244) // conversion from 'boost::compute::ulong_' to '::size_t' [...]
#include <boost/compute/device.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute.hpp>
#pragma warning(pop)

//Dimension of the map
#define N 64 
#define M 64
//Number of obstacles
#define O 32
#define obstacles_tpye 1

namespace compute = boost::compute;

namespace {
// Helper for pritty printing bytes
std::string bytes(unsigned long long bytes) {
    if (bytes > (1 << 20)) // bytes is bigger than 1 MByte
        return std::to_string(bytes >> 20) + " MBytes";
    if (bytes > (1 << 10)) // bytes is bigger than 1 KByte
        return std::to_string(bytes >> 10) + " KBytes";
    return std::to_string(bytes) + " bytes"; // bytes is smaller than 1 KByte
}
} // namespace



void show_map(std::vector<compute::float_> vect) { 
cv::Mat image = cv::Mat::zeros(N, M, CV_8UC1);

    // Fill the matrix with grayscale values
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            image.at<uchar>(i, j) = vect[i*N + j];
        }
    }

    // Display the image
    cv::Mat outImg;
    cv::resize(image, outImg, cv::Size(), 8, 8, cv::INTER_NEAREST); 
    cv::imshow("Grayscale Image", outImg);
    cv::waitKey(0);

}

int main() {

    // Select default OpenCL device
    compute::device dev = compute::system::default_device();
    const auto maxMemAllocSize = dev.get_info<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(); // how much GPU memory is available for allocation
    const auto maxWorkGroupSize = dev.get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>(); // The maximum work group size in the GPU
    const auto maxWorkItemDimensions = dev.get_info<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(); // The maximum dimension of the work item. It should be 3
    const auto maxWorkItemSizes = dev.get_info<CL_DEVICE_MAX_WORK_ITEM_SIZES>(); // The maximum number of work-items for each dimension in work group.

        // print the resutls
        std::cout << "OpenCL device: " << dev.name()
                  << "\n - Compute units: " << dev.compute_units()
                  << "\n - Global memory: " << bytes(dev.global_memory_size())
                  << "\n - Local memory: " << bytes(dev.local_memory_size())
                  << "\n - Max. memory allocation: " << bytes(maxMemAllocSize)
                  << "\n - Max. work group size: " << maxWorkGroupSize
                  << "\n - Max. work item sizes:";
        for (unsigned i = 0; i < maxWorkItemDimensions; ++i)
            std::cout << ' ' << maxWorkItemSizes[i];
        std::cout << std::endl;

        compute::context       context(dev); // object manages memory buffers and program objects
        compute::command_queue queue(context, dev); // Create a queue object to handle the commands for the buffers.
        auto program = compute::program::create_with_source_file("matrixAverageFilter.cl", context); // create a program from a source file: gpuAStar.cl
        program.build();

        const auto localWorkSize =16;//This number result in 16*16=256 max work group size
        const auto globalWorkSize =(std::size_t) std::ceil((double) N/ localWorkSize) * localWorkSize;
        const auto maxLocalBytes = (std::size_t)(dev.local_memory_size() * 0.99);
        const auto perthreadLocalBytes = std::min(
            (std::size_t)(10), (std::size_t)(maxLocalBytes / (localWorkSize * localWorkSize)));
        const auto localMemoryBytes = localWorkSize * perthreadLocalBytes;
        assert(localMemoryBytes <= dev.local_memory_size());
        const auto localMemorySize = localMemoryBytes / sizeof(compute::float_);
        const auto localMemory = compute::local_buffer<compute::float_>(localMemorySize); 




        // Input and output parameters:
        std::vector<compute::float_> inputMatrix;//To be intialized to 0 for obstacles, and 1 otherwise. For goal position inputMatrix=max value
        std::vector<compute::float_> outputMatrix;//To be intialized to zero
        std::vector<compute::float_> kappa;//To be intialized to 0 for obstacles, and 1 otherwise.
        std::vector<compute::float_> intial_map; // To be intialized to 0 for obstacles, and 1 otherwise.

        int start_i = 0;
        int start_j = 0;
        int goal_i = N-1;
        int goal_j = M-1;


       for (int i = 0; i < N; ++i) {
            for (int j=0;j<M;++j) {
                    outputMatrix.emplace_back((float) 0);
                    inputMatrix.emplace_back((float) 0);
                    kappa.emplace_back((float) 1);
                    intial_map.emplace_back((float) 50);
            }
            
       }
       inputMatrix[goal_i * N + goal_j] = 255;
       intial_map[goal_i * N + goal_j] = 255;
       intial_map[start_i * N + start_j] = 255;
       //Obstacles
       std::random_device                 rd;
       std::default_random_engine         generator(rd());
       std::uniform_int_distribution<int> distX(0, N - 1);
       std::uniform_int_distribution<int> distY(0, M - 1);

#if obstacles_tpye == 0
       //////////////////// Random O obstacles
       for (int i = 0; i < O; ++i) {
           int oi = distX(generator);
           int oj = distY(generator);
           if (((oi != start_i) || (oj != start_j)) && ((oi != goal_i) || (oj != goal_j))) {
               inputMatrix[oi * N + oj] = 0;
               kappa[oi * N + oj] = 0;
               intial_map[oi * N + oj] = 0;
           
           }
#elif obstacles_tpye == 1
       
       /////////////// obstacles as a lines in the map
       std::uniform_int_distribution<int> distXo(1, N - 2);
       for (int i = 1; i < N-1; i = i + 2) {
           int oi = distXo(generator);
           if ((i - 1) % 4 == 0)
               for (int j = 0; j < oi; j++) {
                   inputMatrix[i * N + j] = 0;
                   kappa[i * N + j] = 0;
                   intial_map[i * N + j] = 0;
               }
           else
               for (int j = oi; j < M; j++) {
                   inputMatrix[i * N + j] = 0;
                   kappa[i * N + j] = 0;
                   intial_map[i * N + j] = 0;
               }
       }


       show_map(intial_map);
#endif

       //vectors in GPU:
       compute::vector<compute::float_> d_inputMatrix(inputMatrix.size(), context);
       compute::vector<compute::float_> d_outputMatrix(outputMatrix.size(), context);
       compute::vector<compute::float_> d_kappa(outputMatrix.size(), context);

       //set argument for the kernel
       compute::kernel kernel(program, "matrixAverageFilter");      
       kernel.set_arg(0, d_inputMatrix);
       kernel.set_arg(1, d_outputMatrix);
       kernel.set_arg(2, d_kappa);
       kernel.set_arg(3, N);                    
       kernel.set_arg(4, M);
       kernel.set_arg(5, goal_i);
       kernel.set_arg(6, goal_j);
       
       //Copy the parameters to GPU
       compute::copy(inputMatrix.begin(), inputMatrix.end(), d_inputMatrix.begin(),queue);
       compute::copy(outputMatrix.begin(), outputMatrix.end(), d_outputMatrix.begin(),queue);
       compute::copy(kappa.begin(), kappa.end(), d_kappa.begin(), queue);

       std::cout <<"globalWorkSize= "<<globalWorkSize << "\n";
       std::cout << "local_work_size= " << localWorkSize << "\n";
       size_t global_work_offset[2] = {0,0};
       size_t global_work_size[2] = {globalWorkSize, globalWorkSize};
       size_t local_work_size[2] = {localWorkSize, localWorkSize};

       
       std::float_t start_value = 0;
       size_t break_condition = 1;
       
       int  m = 0;
       const auto kernelStart = std::chrono::high_resolution_clock::now();
       while (break_condition>0) {
           queue.enqueue_nd_range_kernel(kernel, 2, global_work_offset, global_work_size,local_work_size);
           queue.finish(); // wait until the kernel finish
           //Copy the start value to the host to check if its value bigger than zero.
           compute::copy(d_outputMatrix.begin() + start_i * N + start_j, d_outputMatrix.begin() + start_i * N + start_j + 1, &start_value, queue);
           if (start_value>0)
           break_condition = 0;
           m++;
           
       }
       queue.finish(); // wait until the kernel finish
       const auto kernelStop = std::chrono::high_resolution_clock::now();

       std::cout << *(inputMatrix.begin() + goal_i * N + goal_j )<< std::endl;
       std::cout << m << std::endl;
       compute::copy(d_outputMatrix.begin(), d_outputMatrix.end(), outputMatrix.begin(), queue);
       
       std::cout << "start_value= " << start_value
                 << std::endl;
       std::cout << "outputMatrix[start_i * N + start_j]= " << outputMatrix[start_i * N + start_j] << std::endl;
       std::cout <<"inputMatrix[goal_i * N + goal_j]= " <<inputMatrix[goal_i * N + goal_j] << std::endl;
       std::cout <<"outputMatrix[goal_i * N + goal_j]= " <<outputMatrix[goal_i * N + goal_j] << std::endl;
       show_map(outputMatrix);


       int i0 = start_i;
       int j0 = start_j;
       std::vector<compute::int_> path_i;
       std::vector<compute::int_> path_j;
       
while (true) {
           path_i.emplace_back((float) i0);
           path_j.emplace_back((float) j0);
           if ((i0 == goal_i) && (j0 == goal_j))
               break;

           float cell_max = outputMatrix[i0 * N + j0];
           int   i_max = i0, j_max = j0;
           for (int ki = -1; ki <= 1; ki++) {
               for (int kj = -1; kj <= 1; kj++) {
                   if ((i0 + ki) >= 0 && (i0 + ki) < N && (j0 + kj) >= 0 && (j0 + kj) < M) {
                       if (outputMatrix[(i0 + ki) * N + (j0 + kj)] >= cell_max) {
                           cell_max = outputMatrix[(i0 + ki) * N + (j0 + kj)];
                           i_max = (i0 + ki);
                           j_max = (j0 + kj);
                       }
                   }
               }
           }
           std::cout << "(i,j)=(" << i0 << "," << j0 << "), V(i,j)=" << outputMatrix[i0 * N + j0];
           std::cout << "V(goal_i,goal_j)=" << outputMatrix[(N - 1) * N + (M - 1)] << std::endl;
           i0 = i_max;
           j0 = j_max;
       }
       for (int i = 0; i < path_i.size(); i++) {
           intial_map[path_i[i] * N + path_j[i]] = 255;
           outputMatrix[path_i[i] * N + path_j[i]] = 255;
       } 


       show_map(outputMatrix);    
       std::cout << "\n - Kernel runtime: "
                 << std::chrono::duration<double>(kernelStop - kernelStart).count() << " seconds"
                 << std::endl;
       std:: cout << "path length=" << path_i.size();
       show_map(intial_map);
    return 0;
}
