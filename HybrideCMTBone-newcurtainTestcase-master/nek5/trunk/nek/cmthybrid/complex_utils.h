
#ifndef COMPLEX_UTILS_H
#define COMPLEX_UTILS_H

// -----------------------------------------------------------------------------

#include <cuComplex.h>
#include <complex>
#include <cmath>
#include <iostream>

// -----------------------------------------------------------------------------

template<typename T>
struct precision_traits
{
    typedef T scalar;
};

template<>
struct precision_traits<cuComplex>
{
    typedef float scalar;
};

template<>
struct precision_traits<cuDoubleComplex>
{
    typedef double scalar;
};

template<typename T>
struct precision_traits< std::complex<T> >
{
    typedef T scalar;
};

// -----------------------------------------------------------------------------

namespace std
{
    // Functions take const& to match the complex versions.

	inline double abs(const cuDoubleComplex& a)
    {
        return std::sqrt(a.x * a.x + a.y * a.y);
    }

	inline float abs(const cuComplex& a)
    {
        return std::sqrt(a.x * a.x + a.y * a.y);
    }

	inline double abs(const std::complex<double>& a)
    {
        return std::sqrt(a.real() * a.real() + a.imag() * a.imag());
    }

	inline float abs(const std::complex<float>& a)
    {
        return std::sqrt(a.real() * a.real() + a.imag() * a.imag());
    }
}

// -----------------------------------------------------------------------------

__host__ __device__
inline cuComplex operator-(const cuComplex& a, const cuComplex& b)
{
	cuComplex ans = { a.x - b.x, a.y - b.y };
	return ans;
}

__host__ __device__
inline cuDoubleComplex operator-(const cuDoubleComplex& a, const cuDoubleComplex& b)
{
	cuDoubleComplex ans = { a.x - b.x, a.y - b.y };
	return ans;
}

__host__ __device__
inline cuComplex operator+(const cuComplex& a, const cuComplex& b)
{
	cuComplex ans = { a.x + b.x, a.y + b.y };
	return ans;
}

__host__ __device__
inline cuDoubleComplex operator+(const cuDoubleComplex& a, const cuDoubleComplex& b)
{
	cuDoubleComplex ans = { a.x + b.x, a.y + b.y };
	return ans;
}

__host__ __device__
inline cuComplex operator-(const cuComplex& a)
{
	cuComplex ans = { - a.x, - a.y};
	return ans;
}

__host__ __device__
inline cuDoubleComplex operator-(const cuDoubleComplex& a)
{
	cuDoubleComplex ans = { - a.x, - a.y};
	return ans;
}

// -----------------------------------------------------------------------------

inline std::ostream& operator<<(std::ostream& os, const cuComplex& a)
{
	os << std::complex<float>(a.x, a.y);
	return os;
}

inline std::ostream& operator<<(std::ostream& os, const cuDoubleComplex& a)
{
	os << std::complex<double>(a.x, a.y);
	return os;
}

// -----------------------------------------------------------------------------

__host__ __device__
inline cuComplex operator*(const cuComplex& a, const cuComplex& b)
{
    // 3 multiplications but more +/-.  Slower.
	//const float t1 = a.x * b.x;
	//const float t2 = a.y * b.y;
	//const float t3 = (a.x + a.y) * (b.x + b.y);
	//cuComplex ret = { t1 - t2, t3 - (t1 + t2) };

	cuComplex ret = { a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x };
	return ret;
}

__host__ __device__
inline cuDoubleComplex operator*(const cuDoubleComplex& a, const cuDoubleComplex& b)
{
    // 3 multiplications but more +/-.  Slower.
	//const double t1 = a.x * b.x;
	//const double t2 = a.y * b.y;
	//const double t3 = (a.x + a.y) * (b.x + b.y);
	//cuDoubleComplex ret = { t1 - t2, t3 - (t1 + t2) };

	cuDoubleComplex ret = { a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x };
	return ret;
}

__host__ __device__
inline void operator+=(cuComplex& a, const cuComplex& b)
{
	a.x += b.x;
	a.y += b.y;
}

__host__ __device__
inline void operator+=(cuDoubleComplex& a, const cuDoubleComplex& b)
{
	a.x += b.x;
	a.y += b.y;
}

__host__ __device__
inline bool operator==(const cuComplex& a, const cuComplex& b)
{
	return a.x == b.x && a.y == b.y;
}

__host__ __device__
inline bool operator==(const cuDoubleComplex& a, const cuDoubleComplex& b)
{
	return a.x == b.x && a.y == b.y;
}

// -----------------------------------------------------------------------------

template<typename T>
__host__ __device__
inline T get_zero()
{
	return T();
}

template<>
__host__ __device__
inline cuComplex get_zero<cuComplex>()
{
	const cuComplex t = {0, 0};
	return t;
}

template<>
__host__ __device__
inline cuDoubleComplex get_zero<cuDoubleComplex>()
{
	const cuDoubleComplex t = {0, 0};
	return t;
}

// -----------------------------------------------------------------------------

template<typename T>
__host__ __device__
inline T get_a(typename precision_traits<T>::scalar v)
{
	return T(v);
}

template<>
__host__ __device__
inline cuComplex get_a<cuComplex>(float v)
{
	const cuComplex t = {v, 0};
	return t;
}

template<>
__host__ __device__
inline cuDoubleComplex get_a<cuDoubleComplex>(double v)
{
	const cuDoubleComplex t = {v, 0};
	return t;
}

// -----------------------------------------------------------------------------

template<typename T>
struct conjugate_unary_func;

template<>
struct conjugate_unary_func<float>
{
	__host__ __device__
	inline float operator()(const float& a) const
	{
		return a;
	}
};

template<>
struct conjugate_unary_func<double>
{
	__host__ __device__
	inline double operator()(const double& a) const
	{
		return a;
	}
};

template<>
struct conjugate_unary_func<cuComplex>
{
	__host__ __device__
	inline cuComplex operator()(const cuComplex& a) const
	{
		cuComplex a_conj = {a.x, -a.y};
		return a_conj;
	}
};

template<>
struct conjugate_unary_func<cuDoubleComplex>
{
	__host__ __device__
	inline cuDoubleComplex operator()(const cuDoubleComplex& a) const
	{
		cuDoubleComplex a_conj = {a.x, -a.y};
		return a_conj;
	}
};

// -----------------------------------------------------------------------------

template<typename T>
struct identity_unary_func
{
	__host__ __device__
	inline T operator()(const T& a) const
	{
		return a;
	}
};

// -----------------------------------------------------------------------------

template<typename T, int a, int b>
struct axpby_ab;

template<typename T> struct axpby_ab<T, -1, -1> { __host__ __device__ inline void operator()(T& y, const T& x) const { y = - x - y; } };
template<typename T> struct axpby_ab<T, -1,  0> { __host__ __device__ inline void operator()(T& y, const T& x) const { y =  -x    ; } };
template<typename T> struct axpby_ab<T, -1,  1> { __host__ __device__ inline void operator()(T& y, const T& x) const { y =  y - x ; } };
template<typename T> struct axpby_ab<T,  0, -1> { __host__ __device__ inline void operator()(T& y, const T& x) const { y = -y     ; } };
template<typename T> struct axpby_ab<T,  0,  0> { __host__ __device__ inline void operator()(T& y, const T& x) const { y =  get_zero<T>(); } };
template<typename T> struct axpby_ab<T,  0,  1> { __host__ __device__ inline void operator()(T& y, const T& x) const { y =  y     ; } };
template<typename T> struct axpby_ab<T,  1, -1> { __host__ __device__ inline void operator()(T& y, const T& x) const { y = x - y  ; } };
template<typename T> struct axpby_ab<T,  1,  0> { __host__ __device__ inline void operator()(T& y, const T& x) const { y = x      ; } };
template<typename T> struct axpby_ab<T,  1,  1> { __host__ __device__ inline void operator()(T& y, const T& x) const { y = x + y  ; } };

// -----------------------------------------------------------------------------

template<typename T, int a>
struct axpby_a;

template<typename T> struct axpby_a<T, -1>
{
	T b; axpby_a(const T& b) : b(b) {}
	__host__ __device__ inline void operator()(T& y, const T& x) const { y = b * y - x; }
};

template<typename T> struct axpby_a<T,  0>
{
	T b; axpby_a(const T& b) : b(b) {}
	__host__ __device__ inline void operator()(T& y, const T& x) const { y = b * y; }
};

template<typename T> struct axpby_a<T,  1>
{
	T b; axpby_a(const T& b) : b(b) {}
	__host__ __device__ inline void operator()(T& y, const T& x) const { y = b * y + x; }
};

// -----------------------------------------------------------------------------

template<typename T, int b>
struct axpby_b;

template<typename T> struct axpby_b<T, -1>
{
	T a; axpby_b(const T& a) : a(a) {}
	__host__ __device__ inline void operator()(T& y, const T& x) const { y = a * x - y; }
};

template<typename T> struct axpby_b<T,  0>
{
	T a; axpby_b(const T& a) : a(a) {}
	__host__ __device__ inline void operator()(T& y, const T& x) const { y = a * x; }
};

template<typename T> struct axpby_b<T,  1>
{
	T a; axpby_b(const T& a) : a(a) {}
	__host__ __device__ inline void operator()(T& y, const T& x) const { y = a * x + y; }
};

// -----------------------------------------------------------------------------

template<typename T>
struct axpby
{
	T a, b; axpby(const T& a, const T& b) : a(a), b(b) {}
	__host__ __device__ inline void operator()(T& y, const T& x) const { y = a * x + b * y; }
};

// -----------------------------------------------------------------------------

#endif // COMPLEX_UTILS_H
