/** Single precision a*x + y kernel */
__kernel void saxpy(__global const float* x, __global float* y, float a)
{
    const int i = get_global_id(0);
    y[i] += a * x[i];
}
