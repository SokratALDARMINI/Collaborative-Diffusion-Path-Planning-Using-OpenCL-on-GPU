__kernel void matrixAverageFilter(__global float* inputMatrix,
                                   __global float* outputMatrix,
                                   __global float* kappa,
                                   const int N,
                                   const int M,
                                   const int goal_i,
                                   const int goal_j)
{
    const int2 matrixSize = (int2)(N, M);
    const int2 pixelCoords = (int2)(get_global_id(0), get_global_id(1));






    float colorAccumulator = 0.0f;
    float kappa_acc = 0.0f;
    for (int k=0;k<100;k++)
    {
    colorAccumulator = 0.0f;
    kappa_acc = 0.0f;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const int2 offset = (int2)(x, y);
            const int2 sampleCoords = pixelCoords + offset;

            if (sampleCoords.x >= 0 && sampleCoords.x < matrixSize.x &&
                sampleCoords.y >= 0 && sampleCoords.y < matrixSize.y) {
                const int index = sampleCoords.y * N + sampleCoords.x;
                colorAccumulator += inputMatrix[index]*kappa[index];
                kappa_acc+=kappa[index];
            }
        }
    }


    float outputColor = colorAccumulator;
    if(kappa_acc!=0)
    outputColor = colorAccumulator / kappa_acc;

    const int outputIndex = pixelCoords.y * N + pixelCoords.x;
    outputMatrix[outputIndex] = outputColor*kappa[outputIndex];
    if(outputIndex==(goal_i*N+goal_j))
    outputMatrix[outputIndex]=255;
    inputMatrix[outputIndex]=outputMatrix[outputIndex];
    

    }
}