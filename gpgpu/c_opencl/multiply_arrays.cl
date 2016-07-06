__kernel void multiply_arrays(__global const float* inputA,
                             __global const float* inputB,
                             __global float* output) {
  int i = get_global_id(0);
  output[i] = inputA[i] * inputB[i];
}
