
/// Single precision a*x + y kernel
__kernel void saxpy(__global float const * const x, __global float* const y, float const a)
{
    size_t const i = get_global_id(0);
    y[i] += a * x[i] * x[i] * exp(y[i] - 1.0f);
}
