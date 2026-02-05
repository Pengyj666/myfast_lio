#ifndef UTILS_COMMON_TIMED_QUEUE_H
#define UTILS_COMMON_TIMED_QUEUE_H

#include <type_traits>
#include <memory>

namespace utils {

template <typename T> 
class TimedQueue {
 public:
  TimedQueue(int maxSize = 0) : m_cap(0), m_sz(0) {
    reset(maxSize);
  }
  ~TimedQueue() {
    release();
  }

  // return: < 0.0 if invalid, >= 0.0 if valid
  double last_timstamp() {
    if (m_sz > 0) {
      return m_Q[(m_tail + m_occ - 1) % m_occ].ts;
    } else {
      return -1.0;
    }
  }

  // get capacity
  int cap() const { return m_cap; }
  // get number of valid elements
  int size() const { return m_sz; }
  // adjust capacity and clear all elements
  bool reset(int maxCap) {
    if (maxCap > 0) {
      if (maxCap != m_cap) {
        m_cap = maxCap;
        m_occ = m_cap + 1;
        m_pData = new char[sizeof(timedData) * m_occ];
        m_Q = (timedData*)m_pData;
        m_head = m_tail = m_sz = 0;
        return (m_pData != nullptr);
      } else {
        m_head = m_tail = m_sz = 0;
        return true;
      }
    }
    return false;
  }
  // adjust capacity if okay and keep all elements
  bool recap(int newMaxCap) {
    if (newMaxCap > 0) {
      if (m_cap > 0 && m_sz > 0) {
        if (newMaxCap > m_sz) {
          char *pNewBuf = new char[sizeof(timedData) * (newMaxCap + 1)];
          if (!pNewBuf) {
            return false;
          }
          timedData *pNewQ = (timedData*)pNewBuf;
          for (int i = m_head, j = 0; i != m_tail; i = (i + 1) % m_occ, ++j) {
            if (std::is_move_constructible<timedData>::value) {
              new (pNewQ + j) timedData(std::move(m_Q[i]));
            } else {
              new (pNewQ + j) timedData(m_Q[i]);
            }
          }
          release();
          m_pData = pNewBuf;
          m_Q = pNewQ;
          m_cap = newMaxCap;
          m_occ = m_cap + 1;
          m_head = 0;
          m_tail = m_sz;
          return true;
        } else {
          return (newMaxCap == m_sz);
        }
      } else {
        return reset(newMaxCap);
      }
    }
    return false;
  }
  
  // emplace a new timed-data into the queue
  bool emplace_back(const T &obj, double timeStamp) {
    if (m_cap > 0) {
      if (m_sz > 0 && timeStamp <= m_Q[(m_tail + m_occ - 1) % m_occ].ts) {
        return false;
      }
      new (&(m_Q[m_tail].obj)) T(obj);
      m_Q[m_tail].ts = timeStamp;
      m_tail = (m_tail + 1) % m_occ;
      if (m_sz == m_cap) {
        (m_Q + m_head)->~timedData();
        m_head = (m_head + 1) % m_occ;
      } else {
        ++m_sz;
      }
      return true;
    }
    return false;
  }
  bool emplace_back(T &&obj, double timeStamp) {
    if (m_cap > 0) {
      if (m_sz > 0 && timeStamp <= m_Q[(m_tail + m_occ - 1) % m_occ].ts) {
        return false;
      }
      new (&(m_Q[m_tail].obj)) T(std::move(obj));
      m_Q[m_tail].ts = timeStamp;
      m_tail = (m_tail + 1) % m_occ;
      if (m_sz == m_cap) {
        (m_Q + m_head)->~timedData();
        m_head = (m_head + 1) % m_occ;
      } else {
        ++m_sz;
      }
      return true;
    }
    return false;
  }
  
  // get reference to the data element
  // index (0 - latested frame, 1 - one frame previous, etc)
  T& operator[] (int index) {
    if (m_cap > 0 && index >= 0 && index < m_sz) {
      return m_Q[(m_tail + m_occ - 1 - index) % m_occ].obj;
    }
    throw "operator[] out-of-bound!";
  }
  const T& operator[] (int index) const {
    if (m_cap > 0 && index >= 0 && index < m_sz) {
      return m_Q[(m_tail + m_occ - 1 - index) % m_occ].obj;
    }
    throw "operator[] out-of-bound!";
  }
  // get time stamp
  // index (0 - latested frame, 1 - one frame previous, etc)
  double& operator() (int index) {
    if (m_cap > 0 && index >= 0 && index < m_sz) {
      return m_Q[(m_tail + m_occ - 1 - index) % m_occ].ts;
    }
    throw "operator() out-of-bound!";
  }
  double operator() (int index) const {
    if (m_cap > 0 && index >= 0 && index < m_sz) {
      return m_Q[(m_tail + m_occ - 1 - index) % m_occ].ts;
    }
    return 0;
  }

  // find an interval to contain timestamp
  // if (0) <= ts2inq return 0;
  // else if (ind) <= ts2inq < (ind-1) return ind;
  // else (out of range) return -1;
  int findAfter(double ts2inq) const {
    if (m_cap > 0 && m_sz > 0) {
      int absInd;
      if (m_head + m_sz > m_occ) { // around the end
        if (ts2inq < m_Q[m_occ - 1].ts) {
          if (!searchRightAfter(ts2inq, m_head, m_occ - 1, &absInd)) {
            return -1;
          }
        } else if (ts2inq >= m_Q[0].ts) {
          if (!searchRightAfter(ts2inq, 0, m_tail - 1, &absInd)) {
            return -1;
          }
        } else {
          absInd = m_occ - 1;
        }
      } else { // consecutive
        if (!searchRightAfter(ts2inq, m_head, (m_tail + m_occ - 1) % m_occ, &absInd)){
          return -1;
        }
      }
      return ((m_tail + m_occ - absInd - 1) % m_occ);
    }
    return -1;
  }

  bool keep(int nFrames2Keep) {
    if (m_cap > 0 && nFrames2Keep >= 0 && nFrames2Keep <= m_sz) {
      for (int i = 0, j = m_head; i < m_sz - nFrames2Keep; ++i, j = (j + 1) % m_occ) {
        (m_Q + j)->~timedData();
      }
      m_head = (m_tail + m_occ - nFrames2Keep) % m_occ;
      m_sz = nFrames2Keep;
      return true;
    }
    return false;
  }

 private:
  int m_cap;
  int m_occ;
  struct timedData {
    double ts;
    T obj;
  };
  char *m_pData;
  timedData *m_Q;
  int m_head, m_tail, m_sz;

  void release() {
    if (m_pData) {
      if (m_sz > 0) {
        for (int i = m_head; i != m_tail; i = (i + 1) % m_occ) {
          (m_Q + i)->~timedData();
        }
      }
      m_pData = nullptr;
    }
  }

  bool searchRightAfter(double ts, int aa, int bb, int *pind) const {
    if (aa > bb || aa < 0 || bb >= m_occ || ts < m_Q[aa].ts) {
      return false;
    }
    if (ts >= m_Q[bb].ts) {
      *pind = bb;
      return true;
    } else if (ts == m_Q[aa].ts) {
      *pind = aa;
      return true;
    }
    while (1) {
      if (bb == aa + 1) {
        *pind = aa;
        return true;
      }
      int cc = (aa + bb) >> 1;
      if (ts > m_Q[cc].ts) {
        aa = cc;
      } else if (ts < m_Q[cc].ts) {
        bb = cc;
      } else {
        *pind = cc;
        return true;
      }
    }
  }
};

} // namespace utils

#endif//UTILS_COMMON_TIMED_QUEUE_H