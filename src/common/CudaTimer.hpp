/* Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef SAMPLES_COMMON_CUDATIMER_HPP__
#define SAMPLES_COMMON_CUDATIMER_HPP__

#include <driver_types.h>
#include <cuda_runtime.h>

namespace dw
{
namespace common
{

class CudaTimer
{
public:
    CudaTimer()
        : m_isTimeValid(false)
        , m_stream(static_cast<cudaStream_t>(0))
    {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }
    ~CudaTimer()
    {
        cudaEventDestroy(m_stop);
        cudaEventDestroy(m_start);
    }

    void setStream(cudaStream_t stream)
    {
        m_stream = stream;
    }

    void start()
    {
        cudaEventRecord(m_start, m_stream);
        m_isTimeValid = false;
    }
    void stop()
    {
        cudaEventRecord(m_stop, m_stream);
        m_isTimeValid = true;
    }

    bool isTimeValid() const
    {
        return m_isTimeValid;
    }

    //Result in us
    float32_t getTime()
    {
        float32_t res;
        cudaEventSynchronize(m_stop);
        cudaEventElapsedTime(&res, m_start, m_stop);
        return 1e3f*res;
    }

private:
    bool m_isTimeValid;
    cudaStream_t m_stream;
    cudaEvent_t m_start;
    cudaEvent_t m_stop;
};

} // namespace testing
} // namespace dw

#endif // TESTS_COMMON_CUDATIMER_HPP__
