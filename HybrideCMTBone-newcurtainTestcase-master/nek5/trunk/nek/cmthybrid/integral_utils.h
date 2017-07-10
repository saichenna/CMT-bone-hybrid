
#ifndef INTEGRAL_UTILS_H
#define INTEGRAL_UTILS_H

#include "static_assert.h"
#include <cassert>

// -----------------------------------------------------------------------------

template<typename index_type, index_type N>
struct integral_utils_is_power_of_2
{
	STATIC_ASSERT(0 < N);

	static const bool value = ((N & (N - 1)) == 0);
};

// -----------------------------------------------------------------------------

template<typename index_type, index_type N>
index_type integral_utils_next_multiple(index_type i)
{
	STATIC_ASSERT(0 < N);

	if(integral_utils_is_power_of_2<index_type, N>::value)
		return (i + (N-1)) & ~(N-1);
	else
		return N*((i + N - 1)/N);
}

// -----------------------------------------------------------------------------

template<typename index_type>
index_type integral_utils_next_multiple(index_type i, index_type N)
{
	// return a multiple of N that is not less than i and closest to it.

	assert(0 < N);
	assert(0 <= i);

	return N*((i + N - 1)/N);
}

// -----------------------------------------------------------------------------

template<typename I, I N>
struct smallest_greater_power_of_2
{
	static const I value = 2 * smallest_greater_power_of_2<I, N/2>::value;
};

template<>
struct smallest_greater_power_of_2<unsigned int, 0>
{
	static const unsigned int value = 1;
};

template<>
struct smallest_greater_power_of_2<int, 0>
{
	static const int value = 1;
};

template<typename I, I N>
struct smallest_greater_or_equal_power_of_2
{
	static const I value = smallest_greater_power_of_2<I, N-1>::value;
};

// -----------------------------------------------------------------------------

template<typename I>
I gcd(I m, I n)
{
	I tmp;

	while(m)
	{
		tmp = m; m = n % m; n = tmp;
	}

	return n;
}

template<typename I>
I lcm(I m, I n)
{
	return m / gcd<I>(m, n) * n;
}

// -----------------------------------------------------------------------------

#endif // INTEGRAL_UTILS_H
