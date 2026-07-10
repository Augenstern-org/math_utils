// tensor.hpp -- cache-line-aligned, SIMD-accelerated, thread-aware 2-D tensor.
//
// Design overview
//   Storage      std::vector<T, AlignedAllocator<T, 64>>. The buffer is 64-byte
//                aligned, satisfying AVX-512 alignment and making cache-line
//                chunk boundaries exact.
//   Copying      Copy construction/assignment deep-copy the buffer. In addition,
//                parallel_copy_from() copies with std::execution::par where the
//                standard library provides parallel algorithms (MSVC STL out of
//                the box; libstdc++ requires TBB), otherwise it falls back to the
//                internal chunked-thread copy.
//   Parallelism  Bulk operations (fill, parallel_apply, add, mul, relu) split the
//                buffer into contiguous chunks whose boundaries are multiples of
//                one cache line (16 floats), one std::thread per chunk, so no two
//                workers ever share a cache line (no false sharing). The thread
//                budget defaults to hardware_concurrency() and can be overridden
//                per tensor with set_num_threads().
//   SIMD         Element-wise float kernels are selected once at runtime via
//                CPUID + XGETBV: AVX-512F when both the CPU and the OS support
//                it, else AVX2, else scalar. Element types other than float use
//                the scalar path. simd_level() exposes the detected level.
//   Locking      One shared_mutex per tensor. Structural changes (resize, being
//                the target of an assignment or parallel_copy_from) take the
//                exclusive lock; size/shape queries, data() and bulk element
//                operations take the shared lock. Bulk operations write elements
//                under the shared lock -- safe against concurrent resize, and
//                worker threads always own disjoint ranges. Two concurrent bulk
//                writes to the *same* tensor from different user threads are not
//                serialized against each other; coordinate that at the call site.
//   Invalidation data()/operator[]/at() results are invalidated by resize() and
//                by assigning into the tensor, exactly like std::vector.
//
// Requires C++17 or later. Depends only on the standard library and, on x86,
// <immintrin.h> plus the compiler's CPUID intrinsics.

#ifndef MATH_UTILS_TENSOR_HPP
#define MATH_UTILS_TENSOR_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define MATH_UTILS_X86 1
#else
#define MATH_UTILS_X86 0
#endif

#if MATH_UTILS_X86
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

// std::execution::par is used only where the standard library actually ships a
// parallel backend: MSVC STL always, libstdc++ only when TBB headers are present.
#if defined(__has_include)
#if __has_include(<version>)
#include <version>
#endif
#endif
#if defined(__cpp_lib_parallel_algorithm) && \
    (defined(_MSC_VER) || (defined(__has_include) && __has_include(<tbb/blocked_range.h>)))
#include <execution>
#define MATH_UTILS_TENSOR_HAS_PAR_ALG 1
#else
#define MATH_UTILS_TENSOR_HAS_PAR_ALG 0
#endif

#if (defined(__GNUC__) || defined(__clang__)) && MATH_UTILS_X86
#define MATH_UTILS_TARGET_AVX2 __attribute__((target("avx2")))
#define MATH_UTILS_TARGET_AVX512 __attribute__((target("avx512f")))
#else
#define MATH_UTILS_TARGET_AVX2
#define MATH_UTILS_TARGET_AVX512
#endif

namespace math_utils {

inline constexpr std::size_t kCacheLineBytes = 64;

// 按照缓存行 64byte 对齐
template <typename T, std::size_t Alignment = kCacheLineBytes>
class AlignedAllocator {
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");
    static_assert(Alignment >= alignof(T), "Alignment must not be weaker than alignof(T)");

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc{};
        }
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
    }

    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t{Alignment});
    }

    friend bool operator==(const AlignedAllocator&, const AlignedAllocator&) noexcept { return true; }
    friend bool operator!=(const AlignedAllocator&, const AlignedAllocator&) noexcept { return false; }
};

enum class SimdLevel { scalar, avx2, avx512 };

constexpr const char* simd_level_name(SimdLevel level) noexcept {
    switch (level) {
        case SimdLevel::avx512: return "AVX-512F";
        case SimdLevel::avx2: return "AVX2";
        default: return "scalar";
    }
}

#if MATH_UTILS_X86
namespace detail {

// MSCV 下查询 cpuid
inline void cpuid_ex(unsigned leaf, unsigned subleaf, unsigned out[4]) {
#if defined(_MSC_VER)
    int r[4];
    __cpuidex(r, static_cast<int>(leaf), static_cast<int>(subleaf));
    for (int i = 0; i < 4; ++i) out[i] = static_cast<unsigned>(r[i]);
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __get_cpuid_count(leaf, subleaf, &a, &b, &c, &d);
    out[0] = a;
    out[1] = b;
    out[2] = c;
    out[3] = d;
#endif
}

inline std::uint64_t xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    // Raw encoding of `xgetbv` so no -mxsave build flag is required.
    std::uint32_t eax, edx;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
}

// AVX-512 requires not only the CPUID feature bit but also OS support for the
// wider register state (XCR0 opmask/ZMM bits), hence the XGETBV checks.
inline SimdLevel detect_simd_level() {
    unsigned r[4] = {0, 0, 0, 0};
    cpuid_ex(0, 0, r);
    const unsigned max_leaf = r[0];
    if (max_leaf < 1) return SimdLevel::scalar;

    cpuid_ex(1, 0, r);
    const bool osxsave = (r[2] & (1u << 27)) != 0;
    const bool avx = (r[2] & (1u << 28)) != 0;
    if (!osxsave || !avx) return SimdLevel::scalar;

    const std::uint64_t xcr0 = xgetbv0();
    const bool ymm_enabled = (xcr0 & 0x06) == 0x06;  // XMM + YMM state
    const bool zmm_enabled = (xcr0 & 0xE6) == 0xE6;  // + opmask, ZMM0-15 high, ZMM16-31
    if (!ymm_enabled || max_leaf < 7) return SimdLevel::scalar;

    cpuid_ex(7, 0, r);
    const bool avx2 = (r[1] & (1u << 5)) != 0;
    const bool avx512f = (r[1] & (1u << 16)) != 0;
    if (avx512f && zmm_enabled) return SimdLevel::avx512;
    if (avx2) return SimdLevel::avx2;
    return SimdLevel::scalar;
}

}  // namespace detail
#endif  // MATH_UTILS_X86

// Detected once, cached for the lifetime of the process.
inline SimdLevel simd_level() noexcept {
#if MATH_UTILS_X86
    static const SimdLevel level = detail::detect_simd_level();
    return level;
#else
    return SimdLevel::scalar;
#endif
}

namespace detail {

using BinaryF32Fn = void (*)(const float*, const float*, float*, std::size_t);
using UnaryF32Fn = void (*)(const float*, float*, std::size_t);

inline void add_f32_scalar(const float* x, const float* y, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i] + y[i];
}

inline void mul_f32_scalar(const float* x, const float* y, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i] * y[i];
}

inline void relu_f32_scalar(const float* x, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

#if MATH_UTILS_X86

MATH_UTILS_TARGET_AVX2
inline void add_f32_avx2(const float* x, const float* y, float* out, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
    for (; i < n; ++i) out[i] = x[i] + y[i];
}

MATH_UTILS_TARGET_AVX2
inline void mul_f32_avx2(const float* x, const float* y, float* out, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
    for (; i < n; ++i) out[i] = x[i] * y[i];
}

MATH_UTILS_TARGET_AVX2
inline void relu_f32_avx2(const float* x, float* out, std::size_t n) {
    const __m256 zero = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_max_ps(_mm256_loadu_ps(x + i), zero));
    }
    for (; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

MATH_UTILS_TARGET_AVX512
inline void add_f32_avx512(const float* x, const float* y, float* out, std::size_t n) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    }
    if (i < n) {
        const __mmask16 tail = static_cast<__mmask16>((1u << (n - i)) - 1u);
        const __m512 vx = _mm512_maskz_loadu_ps(tail, x + i);
        const __m512 vy = _mm512_maskz_loadu_ps(tail, y + i);
        _mm512_mask_storeu_ps(out + i, tail, _mm512_add_ps(vx, vy));
    }
}

MATH_UTILS_TARGET_AVX512
inline void mul_f32_avx512(const float* x, const float* y, float* out, std::size_t n) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    }
    if (i < n) {
        const __mmask16 tail = static_cast<__mmask16>((1u << (n - i)) - 1u);
        const __m512 vx = _mm512_maskz_loadu_ps(tail, x + i);
        const __m512 vy = _mm512_maskz_loadu_ps(tail, y + i);
        _mm512_mask_storeu_ps(out + i, tail, _mm512_mul_ps(vx, vy));
    }
}

MATH_UTILS_TARGET_AVX512
inline void relu_f32_avx512(const float* x, float* out, std::size_t n) {
    const __m512 zero = _mm512_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_max_ps(_mm512_loadu_ps(x + i), zero));
    }
    if (i < n) {
        const __mmask16 tail = static_cast<__mmask16>((1u << (n - i)) - 1u);
        _mm512_mask_storeu_ps(out + i, tail, _mm512_max_ps(_mm512_maskz_loadu_ps(tail, x + i), zero));
    }
}

#endif  // MATH_UTILS_X86

struct F32Kernels {
    BinaryF32Fn add;
    BinaryF32Fn mul;
    UnaryF32Fn relu;
};

// The unified dispatcher: resolved once from the detected SIMD level, then every
// float element-wise operation goes through these function pointers.
inline const F32Kernels& f32_kernels() {
    static const F32Kernels selected = [] {
#if MATH_UTILS_X86
        switch (simd_level()) {
            case SimdLevel::avx512:
                return F32Kernels{&add_f32_avx512, &mul_f32_avx512, &relu_f32_avx512};
            case SimdLevel::avx2:
                return F32Kernels{&add_f32_avx2, &mul_f32_avx2, &relu_f32_avx2};
            default:
                break;
        }
#endif
        return F32Kernels{&add_f32_scalar, &mul_f32_scalar, &relu_f32_scalar};
    }();
    return selected;
}

// Below this element count a single thread is faster than spawning workers.
inline constexpr std::size_t kMinElemsPerThread = 4096;

// Runs fn(lo, hi) over disjoint contiguous chunks of [0, n). Chunk sizes are
// rounded up to `granularity` elements (one cache line), so inter-thread
// boundaries never split a cache line. The calling thread works on chunk 0.
// fn is invoked concurrently and must not throw.
template <typename Fn>
void parallel_chunks(std::size_t n, unsigned max_threads, std::size_t granularity, Fn&& fn) {
    if (n == 0) return;
    const std::size_t cap = max_threads > 0 ? max_threads : 1;
    const std::size_t threads = std::clamp<std::size_t>(n / kMinElemsPerThread, 1, cap);
    std::size_t chunk = (n + threads - 1) / threads;
    chunk = ((chunk + granularity - 1) / granularity) * granularity;
    const std::size_t actual = (n + chunk - 1) / chunk;
    if (actual <= 1) {
        fn(std::size_t{0}, n);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(actual - 1);
    for (std::size_t t = 1; t < actual; ++t) {
        const std::size_t lo = t * chunk;
        const std::size_t hi = std::min(n, lo + chunk);
        workers.emplace_back([&fn, lo, hi] { fn(lo, hi); });
    }
    fn(std::size_t{0}, chunk);
    for (auto& w : workers) w.join();
}

}  // namespace detail

template <typename T>
class Tensor {
    static_assert(std::is_trivially_copyable_v<T>,
                  "Tensor<T> must be trivially_copyable!");

public:
    using value_type = T;
    using storage_type = std::vector<T, AlignedAllocator<T, kCacheLineBytes>>;

    // Elements per cache line: the granularity of inter-thread chunk boundaries
    // (16 for float), preventing false sharing between workers.
    static constexpr std::size_t chunk_granularity =
        (kCacheLineBytes / sizeof(T)) > 0 ? kCacheLineBytes / sizeof(T) : 1;

    Tensor() = default;

    explicit Tensor(std::size_t n) : Tensor(1, n) {}

    Tensor(std::size_t rows, std::size_t cols, T init = T{})
        : rows_(rows), cols_(cols), storage_(rows * cols, init) {}

    Tensor(const Tensor& other) {
        std::shared_lock lock(other.mutex_);
        rows_ = other.rows_;
        cols_ = other.cols_;
        threads_ = other.threads_;
        storage_ = other.storage_;
    }

    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;
        std::unique_lock lhs(mutex_, std::defer_lock);
        std::shared_lock rhs(other.mutex_, std::defer_lock);
        std::lock(lhs, rhs);
        rows_ = other.rows_;
        cols_ = other.cols_;
        threads_ = other.threads_;
        storage_ = other.storage_;
        return *this;
    }

    // Locking a mutex can theoretically throw; treated as fatal here so that
    // moves stay noexcept for container-friendliness.
    Tensor(Tensor&& other) noexcept {
        std::unique_lock lock(other.mutex_);
        rows_ = std::exchange(other.rows_, 0);
        cols_ = std::exchange(other.cols_, 0);
        threads_ = other.threads_;
        storage_ = std::move(other.storage_);
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;
        std::unique_lock lhs(mutex_, std::defer_lock);
        std::unique_lock rhs(other.mutex_, std::defer_lock);
        std::lock(lhs, rhs);
        rows_ = std::exchange(other.rows_, 0);
        cols_ = std::exchange(other.cols_, 0);
        threads_ = other.threads_;
        storage_ = std::move(other.storage_);
        return *this;
    }

    // --- shape -----------------------------------------------------------

    std::size_t size() const {
        std::shared_lock lock(mutex_);
        return storage_.size();
    }

    std::size_t rows() const {
        std::shared_lock lock(mutex_);
        return rows_;
    }

    std::size_t cols() const {
        std::shared_lock lock(mutex_);
        return cols_;
    }

    std::pair<std::size_t, std::size_t> shape() const {
        std::shared_lock lock(mutex_);
        return {rows_, cols_};
    }

    void resize(std::size_t rows, std::size_t cols) {
        std::unique_lock lock(mutex_);
        rows_ = rows;
        cols_ = cols;
        storage_.resize(rows * cols);
    }

    // --- element access ---------------------------------------------------

    // The returned pointer is 64-byte aligned; it is invalidated by resize()
    // and by assignment into this tensor.
    T* data() {
        std::shared_lock lock(mutex_);
        return storage_.data();
    }

    const T* data() const {
        std::shared_lock lock(mutex_);
        return storage_.data();
    }

    // Unchecked, lock-free hot-path accessors.
    T& operator[](std::size_t i) { return storage_[i]; }
    const T& operator[](std::size_t i) const { return storage_[i]; }

    T& operator()(std::size_t r, std::size_t c) { return storage_[r * cols_ + c]; }
    const T& operator()(std::size_t r, std::size_t c) const { return storage_[r * cols_ + c]; }

    // Bounds-checked accessors.
    T& at(std::size_t i) {
        std::shared_lock lock(mutex_);
        if (i >= storage_.size()) throw std::out_of_range("Tensor::at: index out of range");
        return storage_[i];
    }

    const T& at(std::size_t i) const {
        std::shared_lock lock(mutex_);
        if (i >= storage_.size()) throw std::out_of_range("Tensor::at: index out of range");
        return storage_[i];
    }

    T& at(std::size_t r, std::size_t c) {
        std::shared_lock lock(mutex_);
        if (r >= rows_ || c >= cols_) throw std::out_of_range("Tensor::at: index out of range");
        return storage_[r * cols_ + c];
    }

    const T& at(std::size_t r, std::size_t c) const {
        std::shared_lock lock(mutex_);
        if (r >= rows_ || c >= cols_) throw std::out_of_range("Tensor::at: index out of range");
        return storage_[r * cols_ + c];
    }

    // --- initialization ----------------------------------------------------

    void fill(T value) {
        std::shared_lock lock(mutex_);
        T* p = storage_.data();
        detail::parallel_chunks(storage_.size(), threads_, chunk_granularity,
                                [p, value](std::size_t lo, std::size_t hi) { std::fill(p + lo, p + hi, value); });
    }

    void zero() { fill(T{}); }

    // --- bulk operations ----------------------------------------------------

    // Deep-copies src (shape included) into this tensor using the parallel
    // standard algorithm when available, else the internal chunked-thread copy.
    void parallel_copy_from(const Tensor& src) {
        if (this == &src) return;
        std::unique_lock dst_lock(mutex_, std::defer_lock);
        std::shared_lock src_lock(src.mutex_, std::defer_lock);
        std::lock(dst_lock, src_lock);
        rows_ = src.rows_;
        cols_ = src.cols_;
        storage_.resize(src.storage_.size());
#if MATH_UTILS_TENSOR_HAS_PAR_ALG
        std::copy(std::execution::par, src.storage_.begin(), src.storage_.end(), storage_.begin());
#else
        const T* s = src.storage_.data();
        T* d = storage_.data();
        detail::parallel_chunks(storage_.size(), threads_, chunk_granularity,
                                [s, d](std::size_t lo, std::size_t hi) { std::copy(s + lo, s + hi, d + lo); });
#endif
    }

    // Applies op(T&) to every element. Each worker thread owns a contiguous,
    // cache-line-aligned sub-range. op is invoked concurrently: it must be
    // thread-safe (stateless lambdas are ideal) and must not throw.
    template <typename Func>
    void parallel_apply(Func op) {
        static_assert(std::is_invocable_v<Func&, T&>, "op must be callable as op(T&)");
        std::shared_lock lock(mutex_);
        T* p = storage_.data();
        detail::parallel_chunks(storage_.size(), threads_, chunk_granularity,
                                [p, &op](std::size_t lo, std::size_t hi) {
                                    for (std::size_t i = lo; i < hi; ++i) op(p[i]);
                                });
    }

    // Element-wise operations; for T = float these dispatch to the AVX-512 /
    // AVX2 / scalar kernel chosen at runtime, for other types to the scalar path.

    void add(const Tensor& rhs) {
        binary_elementwise(rhs, detail::f32_kernels().add,
                           [](T a, T b) { return static_cast<T>(a + b); });
    }

    void mul(const Tensor& rhs) {
        binary_elementwise(rhs, detail::f32_kernels().mul,
                           [](T a, T b) { return static_cast<T>(a * b); });
    }

    void relu() {
        unary_elementwise(detail::f32_kernels().relu,
                          [](T v) { return v > T{} ? v : T{}; });
    }

    // --- threading ----------------------------------------------------------

    // n == 0 restores the hardware_concurrency() default.
    void set_num_threads(unsigned n) {
        std::unique_lock lock(mutex_);
        threads_ = n > 0 ? n : default_thread_count();
    }

    unsigned num_threads() const {
        std::shared_lock lock(mutex_);
        return threads_;
    }

private:
    static unsigned default_thread_count() {
        const unsigned hw = std::thread::hardware_concurrency();
        return hw > 0 ? hw : 1;
    }

    template <typename ScalarOp>
    void binary_elementwise(const Tensor& rhs,
                            [[maybe_unused]] detail::BinaryF32Fn simd_kernel,
                            [[maybe_unused]] ScalarOp scalar_op) {
        std::shared_lock lhs_lock(mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> rhs_lock;
        if (this == &rhs) {
            lhs_lock.lock();
        } else {
            rhs_lock = std::shared_lock(rhs.mutex_, std::defer_lock);
            std::lock(lhs_lock, rhs_lock);
        }
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
            throw std::invalid_argument("Tensor: shape mismatch in element-wise operation");
        }
        T* a = storage_.data();
        const T* b = rhs.storage_.data();
        const std::size_t n = storage_.size();
        if constexpr (std::is_same_v<T, float>) {
            detail::parallel_chunks(n, threads_, chunk_granularity,
                                    [a, b, simd_kernel](std::size_t lo, std::size_t hi) {
                                        simd_kernel(a + lo, b + lo, a + lo, hi - lo);
                                    });
        } else {
            detail::parallel_chunks(n, threads_, chunk_granularity,
                                    [a, b, &scalar_op](std::size_t lo, std::size_t hi) {
                                        for (std::size_t i = lo; i < hi; ++i) a[i] = scalar_op(a[i], b[i]);
                                    });
        }
    }

    template <typename ScalarOp>
    void unary_elementwise([[maybe_unused]] detail::UnaryF32Fn simd_kernel,
                           [[maybe_unused]] ScalarOp scalar_op) {
        std::shared_lock lock(mutex_);
        T* a = storage_.data();
        const std::size_t n = storage_.size();
        if constexpr (std::is_same_v<T, float>) {
            detail::parallel_chunks(n, threads_, chunk_granularity,
                                    [a, simd_kernel](std::size_t lo, std::size_t hi) {
                                        simd_kernel(a + lo, a + lo, hi - lo);
                                    });
        } else {
            detail::parallel_chunks(n, threads_, chunk_granularity,
                                    [a, &scalar_op](std::size_t lo, std::size_t hi) {
                                        for (std::size_t i = lo; i < hi; ++i) a[i] = scalar_op(a[i]);
                                    });
        }
    }

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    storage_type storage_;
    unsigned threads_ = default_thread_count();
    mutable std::shared_mutex mutex_;
};

}  // namespace math_utils

#endif  // MATH_UTILS_TENSOR_HPP
