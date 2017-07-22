/* Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef SAMPLES_COMMON_STATSCOUNTER_HPP__
#define SAMPLES_COMMON_STATSCOUNTER_HPP__

#include <limits>
#include <sstream>
#include <algorithm>
#include <vector>

namespace dw
{
namespace common
{

// Classes
class StatsCounter
{
  public:
    StatsCounter()
        : m_count(0)
        , m_sum(0)
        , m_sumSq(0)
        , m_max(std::numeric_limits<float32_t>::lowest())
        , m_min(std::numeric_limits<float32_t>::max())
    {
    }

    void addSample(float32_t sample)
    {
        m_samples.push_back(sample);

        m_count++;
        m_sum += sample;
        m_sumSq += sample*sample;
        if (sample > m_max)
            m_max = sample;
        if (sample < m_min)
            m_min = sample;
    }
    void addSample(uint32_t sample)
    {
        addSample(static_cast<float32_t>(sample));
    }
    void addSample(int32_t sample)
    {
        addSample(static_cast<float32_t>(sample));
    }

    template<typename T>
    void addSampleArray(const T *array, uint32_t size)
    {
        for (uint32_t i = 0; i < size; i++)
            addSample(static_cast<float32_t>(array[i]));
    }

    uint32_t getSampleCount() const
    {
        return m_count;
    }

    float32_t getMin() const
    {
        return m_min;
    }

    float32_t getMax() const
    {
        return m_max;
    }

    float32_t getSum() const
    {
        return m_sum;
    }

    float32_t getMean() const
    {
        return m_sum/m_count;
    }

    float32_t getVariance() const
    {
        float32_t mean = getMean();
        return m_sumSq/m_count - mean*mean;
    }

    float32_t getStdDev() const
    {
        return sqrt(getVariance());
    }

    /*
    * Note: To simplify code and make the computation faster, this returns the true median only for odd sample counts.
    *       For even counts the true median would be (samples[n/2] + samples[n/2+1])/2, but sample[n/2] is returned instead.
    */
    float32_t getMedian()
    {
        if (m_samples.empty())
            return 0;

        size_t n = m_samples.size() / 2;
        std::nth_element(m_samples.begin(), m_samples.begin()+n, m_samples.end());
        return m_samples[n];
    }

    /*
     * Note: This method overrides constness to reorder the samples vector. It is considered const because all stats remain the same.
     */
    float32_t getMedian() const
    {
        return const_cast<StatsCounter*>(this)->getMedian();
    }

    template<class TStream>
    void writeToStream(TStream &stream)
    {
        stream << "Median=" << getMedian() << ", mean=" << getMean() << ", var=" << getVariance() << ", std dev=" << getStdDev()
            << ", sample count=" << getSampleCount() << ", min=" << getMin() << ", max=" << getMax();
    }

  protected:
    //Full sample array stored to calculate median
    std::vector<float32_t> m_samples;

    //These are used to calculate mean and variance
    uint32_t m_count;
    float32_t m_sum;
    float32_t m_sumSq;
    
    float32_t m_max;
    float32_t m_min;
};

inline
std::ostream &operator <<(std::ostream &stream, StatsCounter &counter)
{
    counter.writeToStream(stream);
    return stream;
}

} // namespace testing
} // namespace dw

#endif // TESTS_COMMON_STATSCOUNTER_HPP__
