////////////////////////////////////////////////////////////////////////////////
// Crude 2D Lattice Boltzmann Demo program
// CUDA version
// Graham Pullan - Oct 2008
//
// This is a 9 velocity set method:
// Distribution functions are stored as "f" arrays
// Think of these as the number of particles moving in these directions:
//
//      f6  f2   f5
//        \  |  /
//         \ | /
//          \|/
//      f3---|--- f1
//          /|\
//         / | \       and f0 for the rest (zero) velocity
//        /  |  \
//      f7  f4   f8
//
///////////////////////////////////////////////////////////////////////////////

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <vector>

constexpr int TILE_I = 16;
constexpr int TILE_J = 16;

constexpr int I2D(int ni, int i, int j) { return ni * j + i; }

// Check the CUDA errors using the form from talonmies on SO:
// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCuda(ans)                                                                             \
    {                                                                                              \
        cudaAssert((ans), __FILE__, __LINE__);                                                     \
    }

inline void cudaAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// OpenGL pixel buffer object and texture
GLuint gl_PBO, gl_Tex;

// arrays on host //
float *f0, *f1, *f2, *f3, *f4, *f5, *f6, *f7, *f8, *plot;

// rgba arrays for plotting
std::vector<unsigned int> cmap_rgba;

// arrays on device //
float *device_f0, *device_f1, *device_f2, *device_f3, *device_f4;
float *device_f5, *device_f6, *device_f7, *device_f8, *plot_data;

int* device_solid;
std::vector<int> solid;
unsigned int *device_cmap_rgba, *device_plot_rgba;

// textures on device
texture<float, 2> f1_tex, f2_tex, f3_tex, f4_tex, f5_tex, f6_tex, f7_tex, f8_tex;

// CUDA special format arrays on device //
cudaArray *f1_array, *f2_array, *f3_array, *f4_array;
cudaArray *f5_array, *f6_array, *f7_array, *f8_array;

// scalars
float tau;
float vxin, roout;
float width, height;
float minvar, maxvar;

// Some factors used in equilibrium f's
float constexpr faceq1 = 4.0f / 9.0f;
float constexpr faceq2 = 1.0f / 9.0f;
float constexpr faceq3 = 1.0f / 36.0f;

int ni, nj;
int nsolid, nstep, nsteps, ncol;
int ipos_old, jpos_old, draw_solid_flag;

std::size_t pitch;

// CUDA kernel
__global__ void stream_kernel(int const pitch,
                              float* __restrict__ device_f1,
                              float* __restrict__ device_f2,
                              float* __restrict__ device_f3,
                              float* __restrict__ device_f4,
                              float* __restrict__ device_f5,
                              float* __restrict__ device_f6,
                              float* __restrict__ device_f7,
                              float* __restrict__ device_f8)
{
    auto const i = blockIdx.x * TILE_I + threadIdx.x;
    auto const j = blockIdx.y * TILE_J + threadIdx.y;

    auto const index = i + j * pitch / sizeof(float);

    // look up the adjacent f's needed for streaming using textures
    // i.e. gather from textures, write to device memory: device_f1, etc
    device_f1[index] = tex2D(f1_tex, static_cast<float>(i - 1), static_cast<float>(j));
    device_f2[index] = tex2D(f2_tex, static_cast<float>(i), static_cast<float>(j - 1));
    device_f3[index] = tex2D(f3_tex, static_cast<float>(i + 1), static_cast<float>(j));
    device_f4[index] = tex2D(f4_tex, static_cast<float>(i), static_cast<float>(j + 1));
    device_f5[index] = tex2D(f5_tex, static_cast<float>(i - 1), static_cast<float>(j - 1));
    device_f6[index] = tex2D(f6_tex, static_cast<float>(i + 1), static_cast<float>(j - 1));
    device_f7[index] = tex2D(f7_tex, static_cast<float>(i + 1), static_cast<float>(j + 1));
    device_f8[index] = tex2D(f8_tex, static_cast<float>(i - 1), static_cast<float>(j + 1));
}

// CUDA kernel all BC's apart from periodic boundaries:
__global__ void apply_BCs_kernel(int ni,
                                 int nj,
                                 int pitch,
                                 float vxin,
                                 float roout,
                                 float faceq2,
                                 float faceq3,
                                 float* __restrict__ device_f0,
                                 float* __restrict__ device_f1,
                                 float* __restrict__ device_f2,
                                 float* __restrict__ device_f3,
                                 float* __restrict__ device_f4,
                                 float* __restrict__ device_f5,
                                 float* __restrict__ device_f6,
                                 float* __restrict__ device_f7,
                                 float* __restrict__ device_f8,
                                 int* __restrict__ device_solid)
{
    auto const i = blockIdx.x * TILE_I + threadIdx.x;
    auto const j = blockIdx.y * TILE_J + threadIdx.y;

    auto const index = i + j * pitch / sizeof(float);

    // Solid BC: "bounce-back"
    if (device_solid[index] == 0)
    {
        float f1old = device_f1[index];
        float f2old = device_f2[index];
        float f3old = device_f3[index];
        float f4old = device_f4[index];
        float f5old = device_f5[index];
        float f6old = device_f6[index];
        float f7old = device_f7[index];
        float f8old = device_f8[index];

        device_f1[index] = f3old;
        device_f2[index] = f4old;
        device_f3[index] = f1old;
        device_f4[index] = f2old;
        device_f5[index] = f7old;
        device_f6[index] = f8old;
        device_f7[index] = f5old;
        device_f8[index] = f6old;
    }

    // Inlet BC - very crude
    if (i == 0)
    {
        device_f1[index] = roout * faceq2 * (1.0f + 3.0f * vxin + 4.5f * (vxin * vxin));
        device_f5[index] = roout * faceq3 * (1.0f + 3.0f * vxin + 4.5f * (vxin * vxin));
        device_f8[index] = roout * faceq3 * (1.0f + 3.0f * vxin + 4.5f * (vxin * vxin));
    }
    // Exit BC - very crude
    if (i == ni - 1)
    {
        device_f3[index] = device_f3[index - 1];
        device_f6[index] = device_f6[index - 1];
        device_f7[index] = device_f7[index - 1];
    }
}

__global__ void collide_kernel(int const pitch,
                               float const tau,
                               float const faceq1,
                               float const faceq2,
                               float const faceq3,
                               float* const __restrict__ device_f0,
                               float* const __restrict__ device_f1,
                               float* const __restrict__ device_f2,
                               float* const __restrict__ device_f3,
                               float* const __restrict__ device_f4,
                               float* const __restrict__ device_f5,
                               float* const __restrict__ device_f6,
                               float* const __restrict__ device_f7,
                               float* const __restrict__ device_f8,
                               float* const __restrict__ plot_data)
{
    auto const i = blockIdx.x * TILE_I + threadIdx.x;
    auto const j = blockIdx.y * TILE_J + threadIdx.y;

    auto const index = i + j * pitch / sizeof(float);

    // Read all f's and store in registers
    auto const f0now = device_f0[index];
    auto const f1now = device_f1[index];
    auto const f2now = device_f2[index];
    auto const f3now = device_f3[index];
    auto const f4now = device_f4[index];
    auto const f5now = device_f5[index];
    auto const f6now = device_f6[index];
    auto const f7now = device_f7[index];
    auto const f8now = device_f8[index];

    // Macroscopic flow props:
    auto const ro = f0now + f1now + f2now + f3now + f4now + f5now + f6now + f7now + f8now;
    auto const vx = (f1now - f3now + f5now - f6now - f7now + f8now) / ro;
    auto const vy = (f2now - f4now + f5now + f6now - f7now - f8now) / ro;

    // Calculate equilibrium f's
    auto const v_sq_term = 1.5f * (vx * vx + vy * vy);

    auto const f0eq = ro * faceq1 * (1.0f - v_sq_term);
    auto const f1eq = ro * faceq2 * (1.0f + 3.0f * vx + 4.5f * vx * vx - v_sq_term);
    auto const f2eq = ro * faceq2 * (1.0f + 3.0f * vy + 4.5f * vy * vy - v_sq_term);
    auto const f3eq = ro * faceq2 * (1.0f - 3.0f * vx + 4.5f * vx * vx - v_sq_term);
    auto const f4eq = ro * faceq2 * (1.0f - 3.0f * vy + 4.5f * vy * vy - v_sq_term);
    auto const f5eq = ro * faceq3
                      * (1.0f + 3.0f * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) - v_sq_term);
    auto const f6eq = ro * faceq3
                      * (1.0f + 3.0f * (-vx + vy) + 4.5f * (-vx + vy) * (-vx + vy) - v_sq_term);
    auto const f7eq = ro * faceq3
                      * (1.0f + 3.0f * (-vx - vy) + 4.5f * (-vx - vy) * (-vx - vy) - v_sq_term);
    auto const f8eq = ro * faceq3
                      * (1.0f + 3.0f * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) - v_sq_term);

    // Do collisions
    device_f0[index] = (1.0f - 1.0f / tau) * f0now + 1.0f / tau * f0eq;
    device_f1[index] = (1.0f - 1.0f / tau) * f1now + 1.0f / tau * f1eq;
    device_f2[index] = (1.0f - 1.0f / tau) * f2now + 1.0f / tau * f2eq;
    device_f3[index] = (1.0f - 1.0f / tau) * f3now + 1.0f / tau * f3eq;
    device_f4[index] = (1.0f - 1.0f / tau) * f4now + 1.0f / tau * f4eq;
    device_f5[index] = (1.0f - 1.0f / tau) * f5now + 1.0f / tau * f5eq;
    device_f6[index] = (1.0f - 1.0f / tau) * f6now + 1.0f / tau * f6eq;
    device_f7[index] = (1.0f - 1.0f / tau) * f7now + 1.0f / tau * f7eq;
    device_f8[index] = (1.0f - 1.0f / tau) * f8now + 1.0f / tau * f8eq;

    // Set plotting variable to velocity magnitude
    plot_data[index] = sqrtf(vx * vx + vy * vy);
}

__global__ void apply_Periodic_BC_kernel(int ni,
                                         int nj,
                                         int pitch,
                                         float* __restrict__ device_f2,
                                         float* __restrict__ device_f4,
                                         float* __restrict__ device_f5,
                                         float* __restrict__ device_f6,
                                         float* __restrict__ device_f7,
                                         float* __restrict__ device_f8)
{
    auto const i = blockIdx.x * TILE_I + threadIdx.x;
    auto const j = blockIdx.y * TILE_J + threadIdx.y;

    auto const index = i + j * pitch / sizeof(float);

    if (j == 0)
    {
        device_f2[index] = device_f2[i + (nj - 1) * pitch / sizeof(float)];
        device_f5[index] = device_f5[i + (nj - 1) * pitch / sizeof(float)];
        device_f6[index] = device_f6[i + (nj - 1) * pitch / sizeof(float)];
    }
    if (j == nj - 1)
    {
        device_f4[index] = device_f4[i];
        device_f7[index] = device_f7[i];
        device_f8[index] = device_f8[i];
    }
}

// CUDA kernel to fill device_plot_rgba array for plotting
__global__ void get_rgba_kernel(int pitch,
                                int ncol,
                                float minvar,
                                float maxvar,
                                float* plot_data,
                                unsigned int* __restrict__ device_plot_rgba,
                                unsigned int* __restrict__ device_cmap_rgba,
                                int* device_solid)
{
    auto const i = blockIdx.x * TILE_I + threadIdx.x;
    auto const j = blockIdx.y * TILE_J + threadIdx.y;

    auto const index = i + j * pitch / sizeof(float);

    float const frac = (plot_data[index] - minvar) / (maxvar - minvar);
    int icol = static_cast<int>(frac * static_cast<float>(ncol));
    device_plot_rgba[index] = device_solid[index] * device_cmap_rgba[icol];
}

void stream()
{
    // Device-to-device mem-copies to transfer data from linear memory (device_f1)
    // to CUDA format memory (f1_array) so we can use these in textures
    checkCuda(cudaMemcpy2DToArray(f1_array,
                                  0,
                                  0,
                                  (void*)device_f1,
                                  pitch,
                                  sizeof(float) * ni,
                                  nj,
                                  cudaMemcpyDeviceToDevice));
    checkCuda(cudaMemcpy2DToArray(f2_array,
                                  0,
                                  0,
                                  (void*)device_f2,
                                  pitch,
                                  sizeof(float) * ni,
                                  nj,
                                  cudaMemcpyDeviceToDevice));
    checkCuda(cudaMemcpy2DToArray(f3_array,
                                  0,
                                  0,
                                  (void*)device_f3,
                                  pitch,
                                  sizeof(float) * ni,
                                  nj,
                                  cudaMemcpyDeviceToDevice));
    checkCuda(cudaMemcpy2DToArray(f4_array,
                                  0,
                                  0,
                                  (void*)device_f4,
                                  pitch,
                                  sizeof(float) * ni,
                                  nj,
                                  cudaMemcpyDeviceToDevice));
    checkCuda(cudaMemcpy2DToArray(f5_array,
                                  0,
                                  0,
                                  (void*)device_f5,
                                  pitch,
                                  sizeof(float) * ni,
                                  nj,
                                  cudaMemcpyDeviceToDevice));
    checkCuda(cudaMemcpy2DToArray(f6_array,
                                  0,
                                  0,
                                  (void*)device_f6,
                                  pitch,
                                  sizeof(float) * ni,
                                  nj,
                                  cudaMemcpyDeviceToDevice));
    checkCuda(cudaMemcpy2DToArray(f7_array,
                                  0,
                                  0,
                                  (void*)device_f7,
                                  pitch,
                                  sizeof(float) * ni,
                                  nj,
                                  cudaMemcpyDeviceToDevice));
    checkCuda(cudaMemcpy2DToArray(f8_array,
                                  0,
                                  0,
                                  (void*)device_f8,
                                  pitch,
                                  sizeof(float) * ni,
                                  nj,
                                  cudaMemcpyDeviceToDevice));

    // Tell CUDA that we want to use f1_array etc as textures. Also
    // define what type of interpolation we want (nearest point)
    f1_tex.filterMode = cudaFilterModePoint;
    checkCuda(cudaBindTextureToArray(f1_tex, f1_array));

    f2_tex.filterMode = cudaFilterModePoint;
    checkCuda(cudaBindTextureToArray(f2_tex, f2_array));

    f3_tex.filterMode = cudaFilterModePoint;
    checkCuda(cudaBindTextureToArray(f3_tex, f3_array));

    f4_tex.filterMode = cudaFilterModePoint;
    checkCuda(cudaBindTextureToArray(f4_tex, f4_array));

    f5_tex.filterMode = cudaFilterModePoint;
    checkCuda(cudaBindTextureToArray(f5_tex, f5_array));

    f6_tex.filterMode = cudaFilterModePoint;
    checkCuda(cudaBindTextureToArray(f6_tex, f6_array));

    f7_tex.filterMode = cudaFilterModePoint;
    checkCuda(cudaBindTextureToArray(f7_tex, f7_array));

    f8_tex.filterMode = cudaFilterModePoint;
    checkCuda(cudaBindTextureToArray(f8_tex, f8_array));

    dim3 grid = dim3(ni / TILE_I, nj / TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);

    stream_kernel<<<grid, block>>>(pitch,
                                   device_f1,
                                   device_f2,
                                   device_f3,
                                   device_f4,
                                   device_f5,
                                   device_f6,
                                   device_f7,
                                   device_f8);

    checkCuda(cudaUnbindTexture(f1_tex));
    checkCuda(cudaUnbindTexture(f2_tex));
    checkCuda(cudaUnbindTexture(f3_tex));
    checkCuda(cudaUnbindTexture(f4_tex));
    checkCuda(cudaUnbindTexture(f5_tex));
    checkCuda(cudaUnbindTexture(f6_tex));
    checkCuda(cudaUnbindTexture(f7_tex));
    checkCuda(cudaUnbindTexture(f8_tex));
}

void collide()
{
    dim3 grid = dim3(ni / TILE_I, nj / TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);

    collide_kernel<<<grid, block>>>(pitch,
                                    tau,
                                    faceq1,
                                    faceq2,
                                    faceq3,
                                    device_f0,
                                    device_f1,
                                    device_f2,
                                    device_f3,
                                    device_f4,
                                    device_f5,
                                    device_f6,
                                    device_f7,
                                    device_f8,
                                    plot_data);
}

void apply_BCs()
{
    dim3 grid = dim3(ni / TILE_I, nj / TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);

    apply_BCs_kernel<<<grid, block>>>(ni,
                                      nj,
                                      pitch,
                                      vxin,
                                      roout,
                                      faceq2,
                                      faceq3,
                                      device_f0,
                                      device_f1,
                                      device_f2,
                                      device_f3,
                                      device_f4,
                                      device_f5,
                                      device_f6,
                                      device_f7,
                                      device_f8,
                                      device_solid);
}

void apply_Periodic_BC(void)
{
    dim3 grid = dim3(ni / TILE_I, nj / TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);

    apply_Periodic_BC_kernel<<<grid, block>>>(ni,
                                              nj,
                                              pitch,
                                              device_f2,
                                              device_f4,
                                              device_f5,
                                              device_f6,
                                              device_f7,
                                              device_f8);
}

// OpenGL function prototypes
// C wrapper
void get_rgba()
{
    dim3 grid = dim3(ni / TILE_I, nj / TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);

    get_rgba_kernel<<<grid, block>>>(pitch,
                                     ncol,
                                     minvar,
                                     maxvar,
                                     plot_data,
                                     device_plot_rgba,
                                     device_cmap_rgba,
                                     device_solid);
}

// This function is called automatically, over and over again,  by GLUT
void display()
{
    // Set upper and lower limits for plotting
    minvar = 0.0;
    maxvar = 0.2;

    // Do one Lattice Boltzmann step: stream, BC, collide:
    stream();
    apply_Periodic_BC();
    apply_BCs();
    collide();

    // For plotting, map the device_plot_rgba array to the gl_PBO pixel buffer
    cudaGLMapBufferObject((void**)&device_plot_rgba, gl_PBO);

    // Fill the device_plot_rgba array (and the pixel buffer)
    get_rgba();
    cudaGLUnmapBufferObject(gl_PBO);

    // Copy the pixel buffer to the texture, ready to display
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ni, nj, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    // Render one quad to the screen and colour it using our texture
    // i.e. plot our plotvar data to the screen
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(0.0, 0.0, 0.0);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(ni, 0.0, 0.0);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(ni, nj, 0.0);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(0.0, nj, 0.0);
    glEnd();
    glutSwapBuffers();
}

// GLUT resize callback to allow us to change the window size
void resize(int w, int h)
{
    width = w;
    height = h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0., ni, 0., nj, -200., 200.);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// GLUT mouse callback. Left button draws the solid, right button removes solid
void mouse(int button, int state, int x, int y)
{
    if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN))
    {
        draw_solid_flag = 0;
        auto const xx = x;
        auto const yy = y;
        ipos_old = xx / width * ni;
        jpos_old = (height - yy) / height * nj;
    }
    if ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN))
    {
        draw_solid_flag = 1;
        auto const xx = x;
        auto const yy = y;
        ipos_old = xx / width * ni;
        jpos_old = (height - yy) / height * nj;
    }
}

// GLUT call back for when the mouse is moving
// This sets the solid array to draw_solid_flag as set in the mouse callback
// It will draw a staircase line if we move more than one pixel since the
// last callback - that makes the coding a bit cumbersome:
void mouse_motion(int x, int y)
{
    float const xx = x;
    float const yy = y;
    int ipos = (int)(xx / width * (float)ni);
    int jpos = (int)((height - yy) / height * (float)nj);

    int i1 = 0, i2 = 0, j1 = 0, j2 = 0;

    if (ipos <= ipos_old)
    {
        i1 = ipos;
        i2 = ipos_old;
        j1 = jpos;
        j2 = jpos_old;
    }
    else
    {
        i1 = ipos_old;
        i2 = ipos;
        j1 = jpos_old;
        j2 = jpos;
    }

    int jlast = j1;

    for (int i = i1; i <= i2; i++)
    {
        int jnext = 0;
        if (i1 != i2)
        {
            float frac = (float)(i - i1) / (float)(i2 - i1);
            jnext = (int)(frac * (j2 - j1)) + j1;
        }
        else
        {
            jnext = j2;
        }
        if (jnext >= jlast)
        {
            solid[I2D(ni, i, jlast)] = draw_solid_flag;
            for (int j = jlast; j <= jnext; j++)
            {
                solid[I2D(ni, i, j)] = draw_solid_flag;
            }
        }
        else
        {
            solid[I2D(ni, i, jlast)] = draw_solid_flag;
            for (int j = jnext; j <= jlast; j++)
            {
                solid[I2D(ni, i, j)] = draw_solid_flag;
            }
        }
        jlast = jnext;
    }

    // Copy the solid array (host) to the device_solid array (device)
    cudaMemcpy2D((void*)device_solid,
                 pitch,
                 (void*)solid.data(),
                 sizeof(int) * ni,
                 sizeof(int) * ni,
                 nj,
                 cudaMemcpyHostToDevice);
    ipos_old = ipos;
    jpos_old = jpos;
}

int main(int argc, char** argv)
{
    float rcol, gcol, bcol;

    FILE* fp_col;
    cudaChannelFormatDesc desc;

    // The following parameters are usually read from a file, but
    // hard code them for the demo:
    ni = 320 * 8;
    nj = 112 * 16;
    vxin = 0.04;
    roout = 1.0;
    tau = 0.51;

    // Write parameters to screen
    std::cout << "ni = " << ni << "\n"
              << "nj = " << nj << "\n"
              << "vxin = " << vxin << "\n"
              << "roout = " << roout << "\n"
              << "tau = " << tau << "\n";

    // Allocate memory on device
    checkCuda(cudaMallocPitch((void**)&device_f0, &pitch, sizeof(float) * ni, nj));
    checkCuda(cudaMallocPitch((void**)&device_f1, &pitch, sizeof(float) * ni, nj));
    checkCuda(cudaMallocPitch((void**)&device_f2, &pitch, sizeof(float) * ni, nj));
    checkCuda(cudaMallocPitch((void**)&device_f3, &pitch, sizeof(float) * ni, nj));
    checkCuda(cudaMallocPitch((void**)&device_f4, &pitch, sizeof(float) * ni, nj));
    checkCuda(cudaMallocPitch((void**)&device_f5, &pitch, sizeof(float) * ni, nj));
    checkCuda(cudaMallocPitch((void**)&device_f6, &pitch, sizeof(float) * ni, nj));
    checkCuda(cudaMallocPitch((void**)&device_f7, &pitch, sizeof(float) * ni, nj));
    checkCuda(cudaMallocPitch((void**)&device_f8, &pitch, sizeof(float) * ni, nj));
    checkCuda(cudaMallocPitch((void**)&plot_data, &pitch, sizeof(float) * ni, nj));

    checkCuda(cudaMallocPitch((void**)&device_solid, &pitch, sizeof(int) * ni, nj));

    desc = cudaCreateChannelDesc<float>();
    checkCuda(cudaMallocArray(&f1_array, &desc, ni, nj));
    checkCuda(cudaMallocArray(&f2_array, &desc, ni, nj));
    checkCuda(cudaMallocArray(&f3_array, &desc, ni, nj));
    checkCuda(cudaMallocArray(&f4_array, &desc, ni, nj));
    checkCuda(cudaMallocArray(&f5_array, &desc, ni, nj));
    checkCuda(cudaMallocArray(&f6_array, &desc, ni, nj));
    checkCuda(cudaMallocArray(&f7_array, &desc, ni, nj));
    checkCuda(cudaMallocArray(&f8_array, &desc, ni, nj));

    // Allocate memory on host and initialise
    std::vector<float> f0(ni * nj, faceq1 * roout * (1.0f - 1.5f * vxin * vxin));
    std::vector<float> f1(ni * nj,
                          faceq2 * roout
                              * (1.0f + 3.0f * vxin + 4.5f * vxin * vxin - 1.5f * vxin * vxin));
    std::vector<float> f2(ni * nj, faceq2 * roout * (1.0f - 1.5f * vxin * vxin));
    std::vector<float> f3(ni * nj,
                          faceq2 * roout
                              * (1.0f - 3.0f * vxin + 4.5f * vxin * vxin - 1.5f * vxin * vxin));
    std::vector<float> f4(ni * nj, faceq2 * roout * (1.0f - 1.5f * vxin * vxin));
    std::vector<float> f5(ni * nj,
                          faceq3 * roout
                              * (1.0f + 3.0f * vxin + 4.5f * vxin * vxin - 1.5f * vxin * vxin));
    std::vector<float> f6(ni * nj,
                          faceq3 * roout
                              * (1.0f - 3.0f * vxin + 4.5f * vxin * vxin - 1.5f * vxin * vxin));
    std::vector<float> f7(ni * nj,
                          faceq3 * roout
                              * (1.0f - 3.0f * vxin + 4.5f * vxin * vxin - 1.5f * vxin * vxin));
    std::vector<float> f8(ni * nj,
                          faceq3 * roout
                              * (1.0f + 3.0f * vxin + 4.5f * vxin * vxin - 1.5f * vxin * vxin));

    std::vector<float> plot(ni * nj, vxin);

    solid.resize(ni * nj, 1);

    // Read in colourmap data for OpenGL display
    fp_col = fopen("cmap.dat", "r");
    if (fp_col == NULL)
    {
        printf("Error: can't open cmap.dat \n");
        return 1;
    }

    fscanf(fp_col, "%d", &ncol);
    cmap_rgba.resize(ncol);
    checkCuda(cudaMalloc((void**)&device_cmap_rgba, sizeof(unsigned int) * ncol));

    for (int i = 0; i < ncol; i++)
    {
        fscanf(fp_col, "%f%f%f", &rcol, &gcol, &bcol);
        cmap_rgba[i] = ((int)(255.0f) << 24) | // convert colourmap to int
                       ((int)(bcol * 255.0f) << 16) | ((int)(gcol * 255.0f) << 8)
                       | ((int)(rcol * 255.0f) << 0);
    }
    fclose(fp_col);

    // Transfer initial data to device
    checkCuda(cudaMemcpy2D((void*)device_f0,
                           pitch,
                           (void*)f0.data(),
                           sizeof(float) * ni,
                           sizeof(float) * ni,
                           nj,
                           cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D((void*)device_f1,
                           pitch,
                           (void*)f1.data(),
                           sizeof(float) * ni,
                           sizeof(float) * ni,
                           nj,
                           cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D((void*)device_f2,
                           pitch,
                           (void*)f2.data(),
                           sizeof(float) * ni,
                           sizeof(float) * ni,
                           nj,
                           cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D((void*)device_f3,
                           pitch,
                           (void*)f3.data(),
                           sizeof(float) * ni,
                           sizeof(float) * ni,
                           nj,
                           cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D((void*)device_f4,
                           pitch,
                           (void*)f4.data(),
                           sizeof(float) * ni,
                           sizeof(float) * ni,
                           nj,
                           cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D((void*)device_f5,
                           pitch,
                           (void*)f5.data(),
                           sizeof(float) * ni,
                           sizeof(float) * ni,
                           nj,
                           cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D((void*)device_f6,
                           pitch,
                           (void*)f6.data(),
                           sizeof(float) * ni,
                           sizeof(float) * ni,
                           nj,
                           cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D((void*)device_f7,
                           pitch,
                           (void*)f7.data(),
                           sizeof(float) * ni,
                           sizeof(float) * ni,
                           nj,
                           cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D((void*)device_f8,
                           pitch,
                           (void*)f8.data(),
                           sizeof(float) * ni,
                           sizeof(float) * ni,
                           nj,
                           cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D((void*)plot_data,
                           pitch,
                           (void*)plot.data(),
                           sizeof(float) * ni,
                           sizeof(float) * ni,
                           nj,
                           cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D((void*)device_solid,
                           pitch,
                           (void*)solid.data(),
                           sizeof(int) * ni,
                           sizeof(int) * ni,
                           nj,
                           cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy((void*)device_cmap_rgba,
                         (void*)cmap_rgba.data(),
                         sizeof(unsigned int) * ncol,
                         cudaMemcpyHostToDevice));

    // Initialise OpenGL display - use glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(ni, nj);     // Window of ni x nj pixels
    glutInitWindowPosition(50, 50); // Window position
    glutCreateWindow("CUDA 2D LB"); // Window title

    std::cout << "Loading extensions: " << glewGetErrorString(glewInit()) << "\n";
    if (!glewIsSupported("GL_VERSION_2_0 "
                         "GL_ARB_pixel_buffer_object "
                         "GL_EXT_framebuffer_object "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return 1;
    }

    // Set up view
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, ni, 0., nj, -200.0, 200.0);

    // Create texture and bind to gl_Tex
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);            // Generate 2D texture
    glBindTexture(GL_TEXTURE_2D, gl_Tex); // bind to gl_Tex
    // texture properties:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ni, nj, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    std::cout << "Texture created.\n";

    // Create pixel buffer object and bind to gl_PBO
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, pitch * nj, NULL, GL_STREAM_COPY);
    checkCuda(cudaGLRegisterBufferObject(gl_PBO));
    std::cout << "Buffer created.\n";

    std::cout << "Starting GLUT main loop...\n";
    glutDisplayFunc(display);
    glutReshapeFunc(resize);
    glutIdleFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(mouse_motion);

    glutMainLoop();

    return 0;
}
