/*
 * PCG Random Number Generation for C++
 *
 * Copyright 2014-2017 Melissa O'Neill <oneill@pcg-random.org>,
 *                     and the PCG Project contributors.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 * Licensed under the Apache License, Version 2.0 (provided in
 * LICENSE-APACHE.txt and at http://www.apache.org/licenses/LICENSE-2.0)
 * or under the MIT license (provided in LICENSE-MIT.txt and at
 * http://opensource.org/licenses/MIT), at your option. This file may not
 * be copied, modified, or distributed except according to those terms.
 *
 * Distributed on an "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, either
 * express or implied.  See your chosen license for details.
 *
 * For additional information about the PCG random number generation scheme,
 * visit http://www.pcg-random.org/.
 */

/*
 * This code provides a a C++ class that can provide 128-bit (or higher)
 * integers.  To produce 2K-bit integers, it uses two K-bit integers,
 * placed in a union that allowes the code to also see them as four K/2 bit
 * integers (and access them either directly name, or by index).
 *
 * It may seem like we're reinventing the wheel here, because several
 * libraries already exist that support large integers, but most existing
 * libraries provide a very generic multiprecision code, but here we're
 * operating at a fixed size.  Also, most other libraries are fairly
 * heavyweight.  So we use a direct implementation.  Sadly, it's much slower
 * than hand-coded assembly or direct CPU support.
 */

#ifndef PCG_UINT128_V2_HPP_INCLUDED
#define PCG_UINT128_V2_HPP_INCLUDED 1

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <climits>
#include <utility>
#include <initializer_list>
#include <type_traits>
#include <array>
#include <limits>

/*
 * We want to lay the type out the same way that a native type would be laid
 * out, which means we must know the machine's endian, at compile time.
 * This ugliness attempts to do so.
 */

#ifndef PCG_LITTLE_ENDIAN
    #if defined(__BYTE_ORDER__)
        #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            #define PCG_LITTLE_ENDIAN 1
        #elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            #define PCG_LITTLE_ENDIAN 0
        #else
            #error __BYTE_ORDER__ does not match a standard endian, pick a side
        #endif
    #elif __LITTLE_ENDIAN__ || _LITTLE_ENDIAN
        #define PCG_LITTLE_ENDIAN 1
    #elif __BIG_ENDIAN__ || _BIG_ENDIAN
        #define PCG_LITTLE_ENDIAN 0
    #elif __x86_64 || __x86_64__ || _M_X64 || __i386 || __i386__ || _M_IX86
        #define PCG_LITTLE_ENDIAN 1
    #elif __powerpc__ || __POWERPC__ || __ppc__ || __PPC__ \
          || __m68k__ || __mc68000__
        #define PCG_LITTLE_ENDIAN 0
    #else
        #error Unable to determine target endianness
    #endif
#endif

namespace pcg_extras 
{

	// Recent versions of GCC have intrinsics we can use to quickly calculate
	// the number of leading and trailing zeros in a number.  If possible, we
	// use them, otherwise we fall back to old-fashioned bit twiddling to figure
	// them out.

	#ifndef PCG_BITCOUNT_T
		typedef uint8_t bitcount_t;
	#else
		typedef PCG_BITCOUNT_T bitcount_t;
	#endif

		namespace
		{
			/*
			 * Provide some useful helper functions
			 *      * flog2                 floor(log2(x))
			 *      * trailingzeros         number of trailing zero bits
			 */

#ifdef __GNUC__         // Any GNU-compatible compiler supporting C++11 has
			 // some useful intrinsics we can use.

			inline bitcount_t flog2(uint32_t v)
			{
				return 31 - __builtin_clz(v);
			}

			inline bitcount_t trailingzeros(uint32_t v)
			{
				return __builtin_ctz(v);
			}

			inline bitcount_t flog2(uint64_t v)
			{
#if UINT64_MAX == ULONG_MAX
				return 63 - __builtin_clzl(v);
#elif UINT64_MAX == ULLONG_MAX
				return 63 - __builtin_clzll(v);
#else
#error Cannot find a function for uint64_t
#endif
			}

			inline bitcount_t trailingzeros(uint64_t v)
			{
#if UINT64_MAX == ULONG_MAX
				return __builtin_ctzl(v);
#elif UINT64_MAX == ULLONG_MAX
				return __builtin_ctzll(v);
#else
#error Cannot find a function for uint64_t
#endif
			}
#elif _MSC_VER > 1900
#include <intrin.h>
#include <immintrin.h>

			inline bitcount_t flog2(const std::uint32_t& v)
			{
				return 31 - _lzcnt_u32(v);
			}

			inline bitcount_t trailingzeros(const std::uint32_t& v)
			{
				return _tzcnt_u32(v);
			}

			inline bitcount_t flog2(const std::uint64_t& v)
			{
#if UINT64_MAX == ULONG_MAX
				return static_cast<bitcount_t>(63 - _lzcnt_u32(v));
#elif UINT64_MAX == ULLONG_MAX
				return static_cast<bitcount_t>(63 - _lzcnt_u64(v));
#else
#error Cannot find a function for uint64_t
#endif
			}

			inline bitcount_t trailingzeros(const std::uint64_t& v)
			{
#if UINT64_MAX == ULONG_MAX
				return static_cast<bitcount_t>(_tzcnt_u32(v));
#elif UINT64_MAX == ULLONG_MAX
				return static_cast<bitcount_t>(_tzcnt_u64(v));
#else
#error Cannot find a function for uint64_t
#endif
			}

			template<typename T>
			constexpr inline unsigned char addwithcarry(unsigned char& carryin, const T& val1, const T& val2, T& out)
			{
				if constexpr(sizeof(T) = 1) //8 bit
				{
					return _addcarry_u8(carryin, val1, val2, &out);
				}
				else if constexpr(sizeof(T) = 2) // 16 bit
				{
					return _addcarry_u16(carryin, val1, val2, &out);
				}
				else if constexpr(sizeof(T) = 4) //32 bit
				{
					return _addcarryx_u32(carryin, val1, val2, &out);
				}
				else if constexpr(sizeof(T) = 8) //64 bit
				{
					return _addcarryx_u64(carryin, val1, val2, &out);
				}
			};

#else                   // Otherwise, we fall back to bit twiddling
			inline bitcount_t flog2(uint32_t v)
			{
				// Based on code by Eric Cole and Mark Dickinson, which appears at
				// https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn

				static const uint8_t multiplyDeBruijnBitPos[32] = {
				  0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
				  8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
				};

				v |= v >> 1; // first round down to one less than a power of 2
				v |= v >> 2;
				v |= v >> 4;
				v |= v >> 8;
				v |= v >> 16;

				return multiplyDeBruijnBitPos[(uint32_t)(v * 0x07C4ACDDU) >> 27];
			}

			inline bitcount_t trailingzeros(uint32_t v)
			{
				static const uint8_t multiplyDeBruijnBitPos[32] = {
				  0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
				  31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
				};

				return multiplyDeBruijnBitPos[((uint32_t)((v & static_cast<uint32_t>(-static_cast<int32_t>(v))) * 0x077CB531U)) >> 27];
			}

			inline bitcount_t flog2(uint64_t v)
			{
				uint32_t high = v >> 32;
				uint32_t low = uint32_t(v);

				return high ? 32 + flog2(high) : flog2(low);
			}

			inline bitcount_t trailingzeros(uint64_t v)
			{
				uint32_t high = v >> 32;
				uint32_t low = uint32_t(v);

				return low ? trailingzeros(low) : trailingzeros(high) + 32;
			}

#endif

			template <typename UInt>
			inline bitcount_t clog2(const UInt& v)
			{
				return flog2(v) + ((v & (-v)) != v);
			}

			template <typename UInt>
			constexpr inline UInt addwithcarry(const UInt& x, const UInt& y, const bool& carryin, bool& carryout)
			{
#if _MSC_VER > 1900 && __clang__
				UInt res;
				carryout = addwithcarry((unsigned char&)carryin, x, y, res);
				return res;
#else
				const UInt half_result = y + carryin;
				const UInt result = x + half_result;
				carryout = (half_result < y) || (result < x);
				return result;
#endif
			}
		}
	//template <typename UInt>
	//inline UInt subwithcarry(UInt x, UInt y, bool carryin, bool* carryout)
	//{
	//    UInt half_result = y + carryin;
	//    UInt result = x - half_result;
	//    *carryout = (half_result < y) || (result > x);
	//    return result;
	//}

	namespace
	{
	#if PCG_LITTLE_ENDIAN
		constexpr const std::uint8_t lowindex = 0;
		constexpr const std::uint8_t highindex = 1;
	#else
		constexpr const std::uint8_t lowindex = 1;
		constexpr const std::uint8_t highindex = 0;
	#endif

	#if PCG_LITTLE_ENDIAN
	#define PCG_LOOP_HEADER(i)			for (auto i = lowindex; i <= highindex; ++i) 
	#define PCG_LOOP_HEADER_REVERSE(i)  for (auto i = highindex; i >= lowindex;--i) 
	#else
	#define PCG_LOOP_HEADER(i)			for (auto i = lowindex; i >= highindex;--i) 
	#define PCG_LOOP_HEADER_REVERSE(i)  for (auto i = highindex; i <= lowindex; ++i)  
	#endif
	}

	template <typename UInt, typename UIntX2>
	class uint_x2 
	{
		static_assert(2*sizeof(UInt) == sizeof(UIntX2), "You made a programming mistake using this type!");
		UIntX2 da[2] { UIntX2(0U),UIntX2(0U)};

	public:
		constexpr uint_x2() = default;

		constexpr uint_x2(UInt v3, UInt v2, UInt v1, UInt v0)
	#if PCG_LITTLE_ENDIAN
		   : da{ UIntX2(v0) + UIntX2(v1) << 32, UIntX2(v2) + UIntX2(v3) << 32 }
	#else
			: da{ UIntX2(v2) + UIntX2(v3) << 32, UIntX2(v0) + UIntX2(v1) << 32 }
	#endif
		{  // Nothing (else) to do
		};

		constexpr uint_x2(UIntX2 v23, UIntX2 v01)
	#if PCG_LITTLE_ENDIAN
			: da{ v01,v23 }
	#else
			: da{ v23,v01 }
	#endif
		{ // Nothing (else) to do
		};

		template<class Integral,
			typename std::enable_if<(std::is_integral<Integral>::value
				&& sizeof(Integral) <= sizeof(UIntX2))
		>::type* = nullptr>
		constexpr uint_x2(Integral v0)
	#if PCG_LITTLE_ENDIAN
			: da{ UIntX2(v0), UIntX2(0U) }
	#else
			: da{ UIntX2(0U),UIntX2(v0) }
	#endif
		{
			// Nothing (else) to do
		};

		//Narrowing Conversions
		explicit constexpr operator std::int64_t() const
		{
			return static_cast<std::int64_t>(da[lowindex]);
		}
		explicit constexpr operator std::int32_t() const
		{
			return static_cast<std::int32_t>(da[lowindex]);
		}
		explicit constexpr operator std::int16_t() const
		{
			return static_cast<std::int16_t>(da[lowindex]);
		}
		explicit constexpr operator std::int8_t() const
		{
			return static_cast<std::int8_t>(da[lowindex]);;
		}

		//explicit constexpr operator std::uint64_t() const
		//{
		//	return da[startindex];
		//}
		explicit constexpr operator std::uint32_t() const
		{
			return static_cast<std::uint32_t>(da[lowindex]);
		}
		explicit constexpr operator std::uint16_t() const
		{
			return static_cast<std::uint16_t>(da[lowindex]);
		}
		explicit constexpr operator std::uint8_t() const
		{
			return static_cast<std::uint8_t>(da[lowindex]);
		}
		// End of Narrowing Conversions

		typedef typename std::conditional<std::is_same_v<uint64_t,unsigned long>,
										  unsigned long long,
										  unsigned long>::type
				uint_missing_t;

		explicit constexpr operator uint_missing_t() const
		{
			return static_cast<uint_missing_t>(da[lowindex]);
		}

		explicit constexpr operator std::size_t() const
		{
			return static_cast<std::size_t>(da[lowindex]);
		}

		explicit constexpr operator bool() const
		{
			return da[0] || da[1];
		}

		template<typename U, typename V>
		constexpr friend uint_x2<U,V> operator*(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		friend std::pair< uint_x2<U,V>, uint_x2<U,V> >
			divmod(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend uint_x2<U,V> operator+(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend uint_x2<U,V> operator-(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend uint_x2<U,V> operator<<(const uint_x2<U,V>&, const bitcount_t&);

		template<typename U, typename V>
		constexpr friend uint_x2<U,V> operator>>(const uint_x2<U,V>&, const bitcount_t&);

		template<typename U, typename V>
		constexpr friend uint_x2<U,V> operator&(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend uint_x2<U,V> operator|(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend uint_x2<U,V> operator^(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend bool operator==(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend bool operator!=(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend bool operator<(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend bool operator<=(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend bool operator>(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend bool operator>=(const uint_x2<U,V>&, const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend uint_x2<U,V> operator~(const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend uint_x2<U,V> operator-(const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend bitcount_t flog2(const uint_x2<U,V>&);

		template<typename U, typename V>
		constexpr friend bitcount_t trailingzeros(const uint_x2<U,V>&);

		constexpr uint_x2& operator*=(const uint_x2& rhs)
		{
			uint_x2 result = *this * rhs;
			return *this = result;
		}

		uint_x2& operator/=(const uint_x2& rhs)
		{
			uint_x2 result = *this / rhs;
			return *this = result;
		}

		uint_x2& operator%=(const uint_x2& rhs)
		{
			uint_x2 result = *this % rhs;
			return *this = result;
		}

		uint_x2& operator+=(const uint_x2& rhs)
		{
			uint_x2 result = *this + rhs;
			return *this = result;
		}

		uint_x2& operator-=(const uint_x2& rhs)
		{
			uint_x2 result = *this - rhs;
			return *this = result;
		}

		uint_x2& operator&=(const uint_x2& rhs)
		{
			uint_x2 result = *this & rhs;
			return *this = result;
		}

		uint_x2& operator|=(const uint_x2& rhs)
		{
			uint_x2 result = *this | rhs;
			return *this = result;
		}

		uint_x2& operator^=(const uint_x2& rhs)
		{
			uint_x2 result = *this ^ rhs;
			return *this = result;
		}

		uint_x2& operator>>=(bitcount_t shift)
		{
			uint_x2 result = *this >> shift;
			return *this = result;
		}

		uint_x2& operator<<=(bitcount_t shift)
		{
			uint_x2 result = *this << shift;
			return *this = result;
		}

		uint_x2& operator++()
		{
			bool carryout;
			da[lowindex] = addwithcarry(da[lowindex], UIntX2(1), false, carryout);
			da[highindex] += carryout;
			return *this;
		}

		uint_x2 operator++(int)
		{
			uint_x2<UInt, UIntX2> result(*this);
			(++*this);
			return result;
		}

	};

	template<typename U, typename V>
	constexpr bitcount_t flog2(const uint_x2<U,V>& v)
	{
		PCG_LOOP_HEADER_REVERSE(i)	{
			if (v.da[i] == 0)
				 continue;
			return flog2(v.da[i]) + (sizeof(V)*CHAR_BIT)*i;
		}
		abort(); //CHECK: Should we really call abort here?
	}

	template<typename U, typename V>
	constexpr bitcount_t trailingzeros(const uint_x2<U,V>& v)
	{
		PCG_LOOP_HEADER(i) {
			if (v.wa[i] != 0)
				return trailingzeros(v.wa[i]) + (sizeof(U)*CHAR_BIT)*i;
		}
		return (sizeof(U)*CHAR_BIT)*4;
	}

	template <typename UInt, typename UIntX2>
	std::pair< uint_x2<UInt,UIntX2>, uint_x2<UInt,UIntX2> >
		divmod(const uint_x2<UInt,UIntX2>& orig_dividend,
			   const uint_x2<UInt,UIntX2>& divisor)
	{
		// If the dividend is less than the divisor, the answer is always zero.
		// This takes care of boundary cases like 0/x (which would otherwise be
		// problematic because we can't take the log of zero.  (The boundary case
		// of division by zero is undefined.)
		if (orig_dividend < divisor)
			return { uint_x2<UInt,UIntX2>(0UL), orig_dividend };

		auto dividend = orig_dividend;

		auto log2_divisor  = flog2(divisor);
		auto log2_dividend = flog2(dividend);
		// assert(log2_dividend >= log2_divisor);
		bitcount_t logdiff = log2_dividend - log2_divisor;

		constexpr uint_x2<UInt,UIntX2> ONE(1UL);
		if (logdiff == 0)
			return { ONE, dividend - divisor };

		// Now we change the log difference to
		//  floor(log2(divisor)) - ceil(log2(dividend))
		// to ensure that we *underestimate* the result.
		logdiff -= 1;

		uint_x2<UInt,UIntX2> quotient(0UL);

		auto qfactor = ONE << logdiff;
		auto factor  = divisor << logdiff;

		do {
			dividend -= factor;
			quotient += qfactor;
			while (dividend < factor) {
				factor  >>= 1;
				qfactor >>= 1;
			}
		} while (dividend >= divisor);

		return { quotient, dividend };
	}

	template <typename UInt, typename UIntX2>
	uint_x2<UInt,UIntX2> operator/(const uint_x2<UInt,UIntX2>& dividend,
								   const uint_x2<UInt,UIntX2>& divisor)
	{
		return divmod(dividend, divisor).first;
	}

	template <typename UInt, typename UIntX2>
	uint_x2<UInt,UIntX2> operator%(const uint_x2<UInt,UIntX2>& dividend,
								   const uint_x2<UInt,UIntX2>& divisor)
	{
		return divmod(dividend, divisor).second;
	}


	template <typename UInt, typename UIntX2>
	constexpr uint_x2<UInt,UIntX2> operator*(const uint_x2<UInt,UIntX2>& a,
								   const uint_x2<UInt,UIntX2>& b)
	{
		uint_x2<UInt,UIntX2> r = {0U, 0U};
		bool carryin = false;
		bool carryout;
	
		constexpr const bitcount_t shift = sizeof(UInt)*CHAR_BIT;

	#if PCG_LITTLE_ENDIAN
		UIntX2 a0b0 = UIntX2(UInt(a.da[lowindex])) * UIntX2(UInt(b.da[lowindex]));
		r.da[lowindex] = a0b0;

		UIntX2 a1b0 = UIntX2(UInt(a.da[lowindex] >> shift)) * UIntX2(UInt(b.da[lowindex]));
		r.da[lowindex] = addwithcarry(a0b0, a1b0 << shift, carryin, carryout);
		carryin = carryout;
		r.da[highindex] = (a1b0 >> shift) + carryin;

		UIntX2 a0b1 = UIntX2(UInt(a.da[lowindex])) * UIntX2(UInt(b.da[lowindex] >> shift));
		carryin = false;
		r.da[lowindex] = addwithcarry(a0b0, a0b1 << shift, carryin, carryout);
		carryin = carryout;
		r.da[highindex] += (a0b1 >> shift) + r.da[highindex] + carryin;

		UIntX2 a1b1 = UIntX2(UInt(a.da[lowindex] >> shift)) * UIntX2(UInt(b.da[lowindex] >> shift));
		r.da[highindex] += a1b1 + a.da[lowindex] * b.da[highindex] + a.da[highindex] * b.da[lowindex];

		

		return r;
	#else
		static_assert(false, "Not implemented yet!");
	#endif
	}


	template <typename UInt, typename UIntX2>
	constexpr uint_x2<UInt,UIntX2> operator+(const uint_x2<UInt,UIntX2>& a,
								   const uint_x2<UInt,UIntX2>& b)
	{
		uint_x2<UInt, UIntX2> r = { 0U, 0U };

		bool carryin = false;
		bool carryout;
		r.da[lowindex] = addwithcarry(a.da[lowindex], b.da[lowindex], carryin, carryout);
		carryin = carryout;
		r.da[highindex] = addwithcarry(a.da[highindex], b.da[highindex], carryin, carryout);
		return r;
	}



	template <typename UInt, typename UIntX2>
	constexpr uint_x2<UInt,UIntX2> operator-(const uint_x2<UInt,UIntX2>& a,
								   const uint_x2<UInt,UIntX2>& b)
	{
		uint_x2<UInt,UIntX2> r = {0U, 0U};
    
		if (a < b)
		{
		//	r.da[stopindex] = std::numeric_limits<UIntX2>::max() - (a.da[stopindex] - b.da[stopindex]);
		//	r.da[startindex] = std::numeric_limits<UIntX2>::max() - (a.da[startindex] - b.da[startindex]);
			return -(b-a);
		}
		else
		{

			const bool carry = (a.da[lowindex] < b.da[lowindex]);

			r.da[highindex] = a.da[highindex] - b.da[highindex] - carry;

			if (carry)
				r.da[lowindex] = std::numeric_limits<UIntX2>::max() - (a.da[lowindex] - b.da[lowindex]);
			else
				r.da[lowindex] = a.da[lowindex] - b.da[lowindex];
		}
		return r;
	}


	template <typename UInt, typename UIntX2>
	constexpr uint_x2<UInt,UIntX2> operator&(const uint_x2<UInt,UIntX2>& a,
								   const uint_x2<UInt,UIntX2>& b)
	{
		return uint_x2<UInt,UIntX2>(a.da[highindex] & b.da[highindex], a.da[lowindex] & b.da[lowindex]);
	}

	template <typename UInt, typename UIntX2>
	constexpr uint_x2<UInt,UIntX2> operator|(const uint_x2<UInt,UIntX2>& a,
								   const uint_x2<UInt,UIntX2>& b)
	{
		return uint_x2<UInt,UIntX2>(a.da[highindex] | b.da[highindex], a.da[lowindex] | b.da[lowindex]);
	}

	template <typename UInt, typename UIntX2>
	constexpr uint_x2<UInt,UIntX2> operator^(const uint_x2<UInt,UIntX2>& a,
								   const uint_x2<UInt,UIntX2>& b)
	{
		return uint_x2<UInt,UIntX2>(a.da[highindex] ^ b.da[highindex], a.da[lowindex] ^ b.da[lowindex]);
	}

	template <typename UInt, typename UIntX2>
	constexpr uint_x2<UInt,UIntX2> operator~(const uint_x2<UInt,UIntX2>& v)
	{
		return uint_x2<UInt,UIntX2>(~v.da[highindex], ~v.da[lowindex]);
	}

	template <typename UInt, typename UIntX2>
	constexpr uint_x2<UInt,UIntX2> operator-(const uint_x2<UInt,UIntX2>& v)
	{
		return ~v + uint_x2<UInt, UIntX2>(1U);
	}

	template <typename UInt, typename UIntX2>
	constexpr bool operator==(const uint_x2<UInt,UIntX2>& a, const uint_x2<UInt,UIntX2>& b)
	{
		return (a.da[highindex] == b.da[highindex]) && (a.da[lowindex] == b.da[lowindex]);
	}

	template <typename UInt, typename UIntX2>
	constexpr bool operator!=(const uint_x2<UInt,UIntX2>& a, const uint_x2<UInt,UIntX2>& b)
	{
		return !operator==(a,b);
	}


	template <typename UInt, typename UIntX2>
	constexpr bool operator<(const uint_x2<UInt,UIntX2>& a, const uint_x2<UInt,UIntX2>& b)
	{
		return  (a.da[highindex] < b.da[highindex]) ||
			((a.da[highindex] == b.da[highindex]) && (a.da[lowindex] < b.da[lowindex]));
	}

	template <typename UInt, typename UIntX2>
	constexpr bool operator>(const uint_x2<UInt,UIntX2>& a, const uint_x2<UInt,UIntX2>& b)
	{
		return operator<(b,a);
	}

	template <typename UInt, typename UIntX2>
	constexpr bool operator<=(const uint_x2<UInt,UIntX2>& a, const uint_x2<UInt,UIntX2>& b)
	{
		return (!(operator<(b,a)) || operator==(a,b));
	}

	template <typename UInt, typename UIntX2>
	constexpr bool operator>=(const uint_x2<UInt,UIntX2>& a, const uint_x2<UInt,UIntX2>& b)
	{
		return !(operator<(a,b)) || operator==(a, b);
	}

	template <typename UInt, typename UIntX2>
	constexpr uint_x2<UInt,UIntX2> operator<<(const uint_x2<UInt,UIntX2>& v,
									const bitcount_t& shift)
	{
		uint_x2<UInt,UIntX2> r = {0U, 0U};
		constexpr const bitcount_t bits    = sizeof(UIntX2) * CHAR_BIT;
		constexpr const bitcount_t bitmask = (bits - 1);
		const bitcount_t shiftdiv = (shift / bits);
		const bitcount_t shiftmod = (shift & bitmask);

		if (shiftmod) {
			UIntX2 carryover = 0;
	#if PCG_LITTLE_ENDIAN
			for (uint8_t out = shiftdiv, in = 0; out < 2; ++out, ++in) {
	#else
			for (uint8_t out = 2-shiftdiv, in = 2; out != 0; /* dec in loop */) {
				--out, --in;
	#endif
				r.da[out] = (v.da[in] << shiftmod) | carryover;
				carryover = (v.da[in] >> (bits - shiftmod));
			}
		} else {
	#if PCG_LITTLE_ENDIAN
			for (uint8_t out = shiftdiv, in = 0; out < 2; ++out, ++in) {
	#else
			for (uint8_t out = 2-shiftdiv, in = 2; out != 0; /* dec in loop */) {
				--out, --in;
	#endif
				r.da[out] = v.da[in];
			}
		}

		return r;
	}

	template <typename UInt, typename UIntX2>
	constexpr uint_x2<UInt,UIntX2> operator>>(const uint_x2<UInt,UIntX2>& v,
									const bitcount_t& shift)
	{
		uint_x2<UInt,UIntX2> r = {0U, 0U};
		constexpr bitcount_t bits    = sizeof(UIntX2) * CHAR_BIT;
		const bitcount_t bitmask = bits - 1;
		const bitcount_t shiftdiv = shift / bits;
		const bitcount_t shiftmod = shift & bitmask;

		if (shiftmod) {
			UIntX2 carryover = 0;
	#if PCG_LITTLE_ENDIAN
			for (uint8_t out = 2-shiftdiv, in = 2; out != 0; /* dec in loop */) {
				--out, --in;
	#else
			for (uint8_t out = shiftdiv, in = 0; out < 2; ++out, ++in) {
	#endif
				r.da[out] = (v.da[in] >> shiftmod) | carryover;
				carryover = (v.da[in] << (bits - shiftmod));
			}
		} else {
	#if PCG_LITTLE_ENDIAN
			for (uint8_t out = 2-shiftdiv, in = 2; out != 0; /* dec in loop */) {
				--out, --in;
	#else
			for (uint8_t out = shiftdiv, in = 0; out < 4; ++out, ++in) {
	#endif
				r.da[out] = v.da[in];
			}
		}

		return r;
	}


	//Some constexpr static tests
	namespace
	{
		using uint128 = pcg_extras::uint_x2<std::uint32_t, std::uint64_t>;
		static_assert((uint128(4961919U, 434274U) >> 64) == uint128(0U, 4961919U),"Right shift failed!");
		static_assert((uint128(3U, 0) >> 1) == uint128(1U, (1ULL << 63) ), "Right shift failed!");

		static_assert((uint128(5420U, 16551694U) << 64) == uint128(16551694U, 0U), "Left shift failed!");
		static_assert((uint128(0, 3U) << 63) == uint128(1U, (1ULL << 63)), "Left shift failed!");
	}



} // namespace pcg_extras

#endif // PCG_UINT128_HPP_INCLUDED
