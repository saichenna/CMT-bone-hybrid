
#ifndef STATIC_ASSERT_H
#define STATIC_ASSERT_H

// -----------------------------------------------------------------------------

// Example usage:
// STATIC_ASSERT(sizeof(short) == sizeof(unsigned short));

template<bool x>
struct STATIC_ASSERTION_FAIL;

template<>
struct STATIC_ASSERTION_FAIL<true>
{
	enum
	{
		value = 1
	};
};

#define STATIC_ASSERT(exp) typedef char \
	static_assert_typedef[STATIC_ASSERTION_FAIL<(bool)(exp)>::value ]

// -----------------------------------------------------------------------------

#endif // STATIC_ASSERT_H
