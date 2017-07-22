/*
 * Profiler.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */
#include "ProfilerCUDA.hpp"

namespace dw
{
namespace common
{

///////////////////////////////////////////////////////////////////////////////////////
ProfilerCUDA::ProfilerCUDA(dwContextHandle_t ctx)
    : m_ctx(ctx)
    , m_showTotals(false)
    , m_totalsFactor(1)
{
    getThreadData()->setName("main");
}

///////////////////////////////////////////////////////////////////////////////////////
bool ProfilerCUDA::empty()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    for (auto &thread : m_threads)
    {
        auto &threadData = *thread.second;
        for (auto &section : threadData.getRootSection()->getSubsections())
            if (!section.second->empty())
                return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    for(auto &thread : m_threads)
        thread.second->reset();
}

///////////////////////////////////////////////////////////////////////////////////////
ProfilerCUDA::ThreadData::ThreadData(ProfilerCUDA *profiler, std::thread::id id)
    : m_profiler(profiler)
    , m_id(id)
    , m_rootSection(this, NULL,"root")
{
    m_activeSections.push(ActiveTimingData{&m_rootSection, 0, nullptr});
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::setCurrentThreadName(const std::string &name)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        //Search for a thread data with the same name
        for(auto it=m_threads.begin(),end=m_threads.end(); it!=end; ++it)
        {
            ProfilerCUDA::ThreadData &data = *it->second;
            if(data.getName() == name)
            {
                //Match!
                //Update the id
                data.setId(std::this_thread::get_id());

                //Update the thread map
                std::unique_ptr<ThreadData> dataPtr = std::move(it->second);
                m_threads.erase(it);
                m_threads.emplace(data.getId(), std::move(dataPtr));
                return;
            }
        }
    }

    //Nobody with the same name
    auto &data = *getThreadData();
    data.setName(name);
}

///////////////////////////////////////////////////////////////////////////////////////
ProfilerCUDA::ThreadData *ProfilerCUDA::getThreadData()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    auto it=m_threads.find(std::this_thread::get_id());
    if(it==m_threads.end())
    {
        ThreadData *data = new ThreadData(this, std::this_thread::get_id());
        m_threads.insert(std::make_pair(data->getId(),std::unique_ptr<ThreadData>(data)));
        return data;
    }
    else
        return it->second.get();
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::collectTimers()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto &thread : m_threads)
    {
        thread.second->collectTimers();
    }
}

///////////////////////////////////////////////////////////////////////////////////////
CudaTimer *ProfilerCUDA::ThreadData::getTimer()
{
    if (m_freeTimers.empty())
    {
        m_ownedTimers.push_back(std::unique_ptr<CudaTimer>(new CudaTimer()));
        return m_ownedTimers.back().get();
    }
    else
    {
        CudaTimer *timer = m_freeTimers.top();
        m_freeTimers.pop();
        return timer;
    }
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::ThreadData::collectTimers()
{
    collectTimers(&m_rootSection);
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::ThreadData::collectTimers(ProfilerCUDA::SectionData *section)
{
    auto timers = section->collectTimers();
    for (auto timer : timers)
    {
        m_freeTimers.push(timer);
    }

    for (auto &child : section->getSubsections())
    {
        collectTimers(child.second.get());
    }
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::SectionData::addTiming(dwTime_t timeCPU, CudaTimer *timer)
{
    m_statsCPU.addSample(static_cast<float32_t>(timeCPU));
    m_pendingTimers.push_back(timer);
}

///////////////////////////////////////////////////////////////////////////////////////
std::vector<CudaTimer*> ProfilerCUDA::SectionData::collectTimers()
{
    for (auto timer : m_pendingTimers)
        m_statsGPU.addSample(timer->getTime());
    
    return std::move(m_pendingTimers); //This clears the vector and returns it so they can be reused
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::ThreadData::reset()
{
    collectTimers();

    //This clears timing but preserves the keys
    m_rootSection.reset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
ProfilerCUDA::SectionData::SectionData(ProfilerCUDA::ThreadData *threadData, ProfilerCUDA::SectionData *parent, const std::string &key)
    : m_threadData(threadData)
    , m_parent(parent)
    , m_key(key)
{
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::SectionData::reset()
{
    m_statsGPU = StatsCounter();
    m_statsCPU = StatsCounter();
    for(auto it=m_childSections.begin(); it!=m_childSections.end(); it++)
        it->second->reset();
}
}

}
