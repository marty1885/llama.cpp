// rwkv_probe/util.h — shared utilities for the rwkv-probe library.
// (Renamed from common.h to avoid colliding with llama.cpp's common/common.h
//  in local-form #include lookups.)
//
// C++17 (the rest of llama.cpp is locked to C++17 so we can't use std::span).
// This header provides a minimal span shim and a small error-reporting helper.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace rwkv_probe {

// minimal C++17 substitute for std::span<T>. non-owning, contiguous, no bounds
// checking on operator[] (use .at() or check .size() yourself).
template <class T>
class span {
public:
    span() : m_data(nullptr), m_size(0) {}
    span(T * data, std::size_t n) : m_data(data), m_size(n) {}

    template <class Container,
              class = decltype(std::declval<Container &>().data()),
              class = decltype(std::declval<Container &>().size())>
    span(Container & c) : m_data(c.data()), m_size(c.size()) {}

    T *         data()  const { return m_data; }
    std::size_t size()  const { return m_size; }
    bool        empty() const { return m_size == 0; }

    T & operator[](std::size_t i) const { return m_data[i]; }
    T & at(std::size_t i) const {
        if (i >= m_size) {
            throw std::out_of_range("rwkv_probe::span index " + std::to_string(i)
                                    + " >= size " + std::to_string(m_size));
        }
        return m_data[i];
    }

    T * begin() const { return m_data; }
    T * end()   const { return m_data + m_size; }

    span<T> subspan(std::size_t off, std::size_t n) const {
        if (off + n > m_size) {
            throw std::out_of_range("rwkv_probe::span::subspan out of range");
        }
        return span<T>(m_data + off, n);
    }

private:
    T *         m_data;
    std::size_t m_size;
};

// implicit T -> const T conversion via this overload set
template <class T> span<const T> as_const(span<T> s) {
    return span<const T>(s.data(), s.size());
}

// throw with a formatted message; used by the safety checks in StateBuf etc.
[[noreturn]] inline void die(const std::string & msg) {
    throw std::runtime_error("rwkv_probe: " + msg);
}

}  // namespace rwkv_probe
