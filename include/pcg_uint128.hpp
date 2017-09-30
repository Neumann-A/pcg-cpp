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
 * Due to constexpr compilation errors with VS 2017 the pcg 128-bit code was replaced
 * by the one from googles abseil library. The uint128 type of abseil was changed to
 * make it mostly constexpr and header only. The following changes where made:
 *	- most functions are now declared constexpr and inline
 *	  (This was needed to make pcg compile with VS 2017; 
 *		with clang-cl there were no issues)
 *	- the abseil makros needed for uint128 have been copied into this file
 *  - functions declared in the int128.hpp have also been moved into this file
 *	- there are also smaller other changes which i dont remember
 *
 * Alexander Neumann
 */

#ifndef PCG_UINT128_HPP_INCLUDED
#define PCG_UINT128_HPP_INCLUDED 1

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <climits>
#include <iosfwd>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace pcg_extras {

//
// Copyright 2017 The Abseil Authors and modified by Alexander Neumann
// for usage in pcg random. Fixed compilation issues with Visual Studio 2017
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//


// ABSL_IS_LITTLE_ENDIAN
// ABSL_IS_BIG_ENDIAN
//
// Checks the endianness of the platform.
//
// Notes: uses the built in endian macros provided by GCC (since 4.6) and
// Clang (since 3.2); see
// https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html.
// Otherwise, if _WIN32, assume little endian. Otherwise, bail with an error.
#if defined(ABSL_IS_BIG_ENDIAN)
#error "ABSL_IS_BIG_ENDIAN cannot be directly set."
#endif
#if defined(ABSL_IS_LITTLE_ENDIAN)
#error "ABSL_IS_LITTLE_ENDIAN cannot be directly set."
#endif

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define ABSL_IS_LITTLE_ENDIAN 1
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define ABSL_IS_BIG_ENDIAN 1
#elif defined(_WIN32)
#define ABSL_IS_LITTLE_ENDIAN 1
#else
#error "absl endian detection needs to be set up for your compiler"
#endif


// ABSL_HAVE_INTRINSIC_INT128
//
// Checks whether the __int128 compiler extension for a 128-bit integral type is
// supported.
//
// Notes: __SIZEOF_INT128__ is defined by Clang and GCC when __int128 is
// supported, except on ppc64 and aarch64 where __int128 exists but has exhibits
// a sporadic compiler crashing bug. Nvidia's nvcc also defines __GNUC__ and
// __SIZEOF_INT128__ but not all versions actually support __int128.
#ifdef ABSL_HAVE_INTRINSIC_INT128
#error ABSL_HAVE_INTRINSIC_INT128 cannot be directly set
#elif (defined(__clang__) && defined(__SIZEOF_INT128__) &&               \
       !defined(__ppc64__) && !defined(__aarch64__)) ||                  \
    (defined(__CUDACC__) && defined(__SIZEOF_INT128__) &&                \
     __CUDACC_VER__ >= 70000) ||                                         \
    (!defined(__clang__) && !defined(__CUDACC__) && defined(__GNUC__) && \
     defined(__SIZEOF_INT128__))
#define ABSL_HAVE_INTRINSIC_INT128 1
#endif

// -----------------------------------------------------------------------------
// (File: int128.h)
// -----------------------------------------------------------------------------
//
// This header file defines 128-bit integer types. Currently, this file defines
// `uint128`, an unsigned 128-bit integer; a signed 128-bit integer is
// forthcoming.

// uint128
//
// An unsigned 128-bit integer type. The API is meant to mimic an intrinsic type
// as closely as is practical, including exhibiting undefined behavior in
// analogous cases (e.g. division by zero). This type is intended to be a
// drop-in replacement once C++ supports an intrinsic `uint128_t` type; when
// that occurs, existing uses of `uint128` will continue to work using that new
// type.
//
// Note: code written with this type will continue to compile once `unint128_t`
// is introduced, provided the replacement helper functions
// `Uint128(Low|High)64()` and `MakeUint128()` are made.
//
// A `uint128` supports the following:
//
//   * Implicit construction from integral types
//   * Explicit conversion to integral types
//
// Additionally, if your compiler supports `__int128`, `uint128` is
// interoperable with that type. (Abseil checks for this compatibility through
// the `ABSL_HAVE_INTRINSIC_INT128` macro.)
//
// However, a `uint128` differs from intrinsic integral types in the following
// ways:
//
//   * Errors on implicit conversions that does not preserve value (such as
//     loss of precision when converting to float values).
//   * Requires explicit construction from and conversion to floating point
//     types.
//   * Conversion to integral types requires an explicit static_cast() to
//     mimic use of the `-Wnarrowing` compiler flag.
//
// Example:
//
//     float y = kuint128max; // Error. uint128 cannot be implicitly converted
//                            // to float.
//
//     uint128 v;
//     uint64_t i = v                          // Error
//     uint64_t i = static_cast<uint64_t>(v)   // OK
//
class alignas(16) uint128 {
 public:
  constexpr uint128() : lo_(0), hi_(0) {};

  // Constructors from arithmetic types
  constexpr uint128(int v);                 // NOLINT(runtime/explicit)
  constexpr uint128(unsigned int v);        // NOLINT(runtime/explicit)
  constexpr uint128(long v);                // NOLINT(runtime/int)
  constexpr uint128(unsigned long v);       // NOLINT(runtime/int)
  constexpr uint128(long long v);           // NOLINT(runtime/int)
  constexpr uint128(unsigned long long v);  // NOLINT(runtime/int)
#ifdef ABSL_HAVE_INTRINSIC_INT128
  constexpr uint128(__int128 v);           // NOLINT(runtime/explicit)
  constexpr uint128(unsigned __int128 v);  // NOLINT(runtime/explicit)
#endif  // ABSL_HAVE_INTRINSIC_INT128
  explicit uint128(float v);        // NOLINT(runtime/explicit)
  explicit uint128(double v);       // NOLINT(runtime/explicit)
  explicit uint128(long double v);  // NOLINT(runtime/explicit)

  // Assignment operators from arithmetic types
  constexpr uint128& operator=(int v);
  constexpr uint128& operator=(unsigned int v);
  constexpr uint128& operator=(long v);                // NOLINT(runtime/int)
  constexpr uint128& operator=(unsigned long v);       // NOLINT(runtime/int)
  constexpr uint128& operator=(long long v);           // NOLINT(runtime/int)
  constexpr uint128& operator=(unsigned long long v);  // NOLINT(runtime/int)
#ifdef ABSL_HAVE_INTRINSIC_INT128
  constexpr uint128& operator=(__int128 v);
  constexpr uint128& operator=(unsigned __int128 v);
#endif  // ABSL_HAVE_INTRINSIC_INT128

  // Conversion operators to other arithmetic types
  constexpr explicit operator bool() const;
  constexpr explicit operator char() const;
  constexpr explicit operator signed char() const;
  constexpr explicit operator unsigned char() const;
  constexpr explicit operator char16_t() const;
  constexpr explicit operator char32_t() const;
  constexpr explicit operator wchar_t() const;
  constexpr explicit operator short() const;  // NOLINT(runtime/int)
  // NOLINTNEXTLINE(runtime/int)
  constexpr explicit operator unsigned short() const;
  constexpr explicit operator int() const;
  constexpr explicit operator unsigned int() const;
  constexpr explicit operator long() const;  // NOLINT(runtime/int)
  // NOLINTNEXTLINE(runtime/int)
  constexpr explicit operator unsigned long() const;
  // NOLINTNEXTLINE(runtime/int)
  constexpr explicit operator long long() const;
  // NOLINTNEXTLINE(runtime/int)
  constexpr explicit operator unsigned long long() const;
#ifdef ABSL_HAVE_INTRINSIC_INT128
  constexpr explicit operator __int128() const;
  constexpr explicit operator unsigned __int128() const;
#endif  // ABSL_HAVE_INTRINSIC_INT128
  explicit operator float() const;
  explicit operator double() const;
  explicit operator long double() const;

  // Trivial copy constructor, assignment operator and destructor.

  // Arithmetic operators.
  constexpr uint128& operator+=(const uint128& other);
  constexpr uint128& operator-=(const uint128& other);
  constexpr uint128& operator*=(const uint128& other);
  // Long division/modulo for uint128.
  constexpr uint128& operator/=(const uint128& other);
  constexpr uint128& operator%=(const uint128& other);
  constexpr uint128 operator++(int);
  constexpr uint128 operator--(int);
  constexpr uint128& operator<<=(int);
  constexpr uint128& operator>>=(int);
  constexpr uint128& operator&=(const uint128& other);
  constexpr uint128& operator|=(const uint128& other);
  constexpr uint128& operator^=(const uint128& other);
  constexpr uint128& operator++();
  constexpr uint128& operator--();

  // Uint128Low64()
  //
  // Returns the lower 64-bit value of a `uint128` value.
  friend inline constexpr uint64_t Uint128Low64(const uint128& v);

  // Uint128High64()
  //
  // Returns the higher 64-bit value of a `uint128` value.
  friend inline constexpr uint64_t Uint128High64(const uint128& v);

  // MakeUInt128()
  //
  // Constructs a `uint128` numeric value from two 64-bit unsigned integers.
  // Note that this factory function is the only way to construct a `uint128`
  // from integer values greater than 2^64.
  //
  // Example:
  //
  //   absl::uint128 big = absl::MakeUint128(1, 0);
  friend inline constexpr uint128 MakeUint128(const uint64_t& top,const uint64_t& bottom);
  inline constexpr uint128(uint64_t top, uint64_t bottom) : hi_(top), lo_(bottom) {};

  private:
  // TODO(strel) Update implementation to use __int128 once all users of
  // uint128 are fixed to not depend on alignof(uint128) == 8. Also add
  // alignas(16) to class definition to keep alignment consistent across
  // platforms.
#if defined(ABSL_IS_LITTLE_ENDIAN)
  uint64_t lo_;
  uint64_t hi_;
#elif defined(ABSL_IS_BIG_ENDIAN)
  uint64_t hi_;
  uint64_t lo_;
#else  // byte order
#error "Unsupported byte order: must be little-endian or big-endian."
#endif  // byte order
};



// allow uint128 to be logged
//extern std::ostream& operator<<(std::ostream& o, const uint128& b);

// TODO(strel) add operator>>(std::istream&, uint128&)

// Methods to access low and high pieces of 128-bit value.
inline constexpr uint64_t Uint128Low64(const uint128& v);
inline constexpr uint64_t Uint128High64(const uint128& v);

// TODO(absl-team): Implement signed 128-bit type

// --------------------------------------------------------------------------
//                      Implementation details follow
// --------------------------------------------------------------------------


inline constexpr uint128 MakeUint128(const uint64_t& top,const uint64_t& bottom) {
	return uint128(top, bottom);
}

static constexpr const auto kuint128max = MakeUint128(std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max());

// Assignment from integer types.

inline constexpr uint128& uint128::operator=(int v) {
  return *this = uint128(v);
}

inline constexpr uint128& uint128::operator=(unsigned int v) {
  return *this = uint128(v);
}

inline constexpr uint128& uint128::operator=(long v) {  // NOLINT(runtime/int)
  return *this = uint128(v);
}

// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128& uint128::operator=(unsigned long v) {
  return *this = uint128(v);
}

// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128& uint128::operator=(long long v) {
  return *this = uint128(v);
}

// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128& uint128::operator=(unsigned long long v) {
  return *this = uint128(v);
}

#ifdef ABSL_HAVE_INTRINSIC_INT128
inline constexpr uint128& uint128::operator=(__int128 v) {
  return *this = uint128(v);
}

inline constexpr uint128& uint128::operator=(unsigned __int128 v) {
  return *this = uint128(v);
}
#endif  // ABSL_HAVE_INTRINSIC_INT128

// Shift and arithmetic operators.

inline constexpr uint128 operator<<(const uint128& lhs, int amount) {
  return uint128(lhs) <<= amount;
}

inline constexpr uint128 operator>>(const uint128& lhs, int amount) {
  return uint128(lhs) >>= amount;
}

inline constexpr uint128 operator+(const uint128& lhs, const uint128& rhs) {
  return uint128(lhs) += rhs;
}

inline constexpr uint128 operator-(const uint128& lhs, const uint128& rhs) {
  return uint128(lhs) -= rhs;
}

inline constexpr uint128 operator*(const uint128& lhs, const uint128& rhs) {
  return uint128(lhs) *= rhs;
}

inline constexpr uint128 operator/(const uint128& lhs, const uint128& rhs) {
  return uint128(lhs) /= rhs;
}

inline constexpr uint128 operator%(const uint128& lhs, const uint128& rhs) {
  return uint128(lhs) %= rhs;
}

inline constexpr uint64_t Uint128Low64(const uint128& v) { return v.lo_; }

inline constexpr uint64_t Uint128High64(const uint128& v) { return v.hi_; }

// Constructors from integer types.

#if defined(ABSL_IS_LITTLE_ENDIAN)

//inline constexpr uint128::uint128(uint64_t top, uint64_t bottom)
//    : lo_(bottom), hi_(top) {}

inline constexpr uint128::uint128(int v)
    : lo_(v), hi_(v < 0 ? std::numeric_limits<uint64_t>::max() : 0) {}
inline constexpr uint128::uint128(long v)  // NOLINT(runtime/int)
    : lo_(v), hi_(v < 0 ? std::numeric_limits<uint64_t>::max() : 0) {}
inline constexpr uint128::uint128(long long v)  // NOLINT(runtime/int)
    : lo_(v), hi_(v < 0 ? std::numeric_limits<uint64_t>::max() : 0) {}

inline constexpr uint128::uint128(unsigned int v) : lo_(v), hi_(0) {}
// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128::uint128(unsigned long v) : lo_(v), hi_(0) {}
// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128::uint128(unsigned long long v)
    : lo_(v), hi_(0) {}

#ifdef ABSL_HAVE_INTRINSIC_INT128
inline constexpr uint128::uint128(__int128 v)
    : lo_(static_cast<uint64_t>(v & ~uint64_t{0})),
      hi_(static_cast<uint64_t>(static_cast<unsigned __int128>(v) >> 64)) {}
inline constexpr uint128::uint128(unsigned __int128 v)
    : lo_(static_cast<uint64_t>(v & ~uint64_t{0})),
      hi_(static_cast<uint64_t>(v >> 64)) {}
#endif  // ABSL_HAVE_INTRINSIC_INT128

#elif defined(ABSL_IS_BIG_ENDIAN)

inline constexpr uint128::uint128(uint64_t top, uint64_t bottom)
    : hi_(top), lo_(bottom) {}

inline constexpr uint128::uint128(int v)
    : hi_(v < 0 ? std::numeric_limits<uint64_t>::max() : 0), lo_(v) {}
inline constexpr uint128::uint128(long v)  // NOLINT(runtime/int)
    : hi_(v < 0 ? std::numeric_limits<uint64_t>::max() : 0), lo_(v) {}
inline constexpr uint128::uint128(long long v)  // NOLINT(runtime/int)
    : hi_(v < 0 ? std::numeric_limits<uint64_t>::max() : 0), lo_(v) {}

inline constexpr uint128::uint128(unsigned int v) : hi_(0), lo_(v) {}
// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128::uint128(unsigned long v) : hi_(0), lo_(v) {}
// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128::uint128(unsigned long long v)
    : hi_(0), lo_(v) {}

#ifdef ABSL_HAVE_INTRINSIC_INT128
inline constexpr uint128::uint128(__int128 v)
    : hi_(static_cast<uint64_t>(static_cast<unsigned __int128>(v) >> 64)),
      lo_(static_cast<uint64_t>(v & ~uint64_t{0})) {}
inline constexpr uint128::uint128(unsigned __int128 v)
    : hi_(static_cast<uint64_t>(v >> 64)),
      lo_(static_cast<uint64_t>(v & ~uint64_t{0})) {}
#endif  // ABSL_HAVE_INTRINSIC_INT128

#else  // byte order
#error "Unsupported byte order: must be little-endian or big-endian."
#endif  // byte order

// Conversion operators to integer types.

inline constexpr uint128::operator bool() const {
  return lo_ || hi_;
}

inline constexpr uint128::operator char() const {
  return static_cast<char>(lo_);
}

inline constexpr uint128::operator signed char() const {
  return static_cast<signed char>(lo_);
}

inline constexpr uint128::operator unsigned char() const {
  return static_cast<unsigned char>(lo_);
}

inline constexpr uint128::operator char16_t() const {
  return static_cast<char16_t>(lo_);
}

inline constexpr uint128::operator char32_t() const {
  return static_cast<char32_t>(lo_);
}

inline constexpr uint128::operator wchar_t() const {
  return static_cast<wchar_t>(lo_);
}

// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128::operator short() const {
  return static_cast<short>(lo_);  // NOLINT(runtime/int)
}

// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128::operator unsigned short() const {
  return static_cast<unsigned short>(lo_);  // NOLINT(runtime/int)
}

inline constexpr uint128::operator int() const {
  return static_cast<int>(lo_);
}

inline constexpr uint128::operator unsigned int() const {
  return static_cast<unsigned int>(lo_);
}

// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128::operator long() const {
  return static_cast<long>(lo_);  // NOLINT(runtime/int)
}

// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128::operator unsigned long() const {
  return static_cast<unsigned long>(lo_);  // NOLINT(runtime/int)
}

// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128::operator long long() const {
  return static_cast<long long>(lo_);  // NOLINT(runtime/int)
}

// NOLINTNEXTLINE(runtime/int)
inline constexpr uint128::operator unsigned long long() const {
  return static_cast<unsigned long long>(lo_);  // NOLINT(runtime/int)
}

#ifdef ABSL_HAVE_INTRINSIC_INT128
inline constexpr uint128::operator __int128() const {
  return (static_cast<__int128>(hi_) << 64) + lo_;
}

inline constexpr uint128::operator unsigned __int128() const {
  return (static_cast<unsigned __int128>(hi_) << 64) + lo_;
}
#endif  // ABSL_HAVE_INTRINSIC_INT128

// Conversion operators to floating point types.

inline uint128::operator float() const {
  return static_cast<float>(lo_) + std::ldexp(static_cast<float>(hi_), 64);
}

inline uint128::operator double() const {
  return static_cast<double>(lo_) + std::ldexp(static_cast<double>(hi_), 64);
}

inline uint128::operator long double() const {
  return static_cast<long double>(lo_) +
         std::ldexp(static_cast<long double>(hi_), 64);
}

// Comparison operators.

inline constexpr bool operator==(const uint128& lhs, const uint128& rhs) {
  return (Uint128Low64(lhs) == Uint128Low64(rhs) &&
          Uint128High64(lhs) == Uint128High64(rhs));
}

inline constexpr bool operator!=(const uint128& lhs, const uint128& rhs) {
  return !(lhs == rhs);
}

inline constexpr bool operator<(const uint128& lhs, const uint128& rhs) {
  return (Uint128High64(lhs) == Uint128High64(rhs))
             ? (Uint128Low64(lhs) < Uint128Low64(rhs))
             : (Uint128High64(lhs) < Uint128High64(rhs));
}

inline constexpr bool operator>(const uint128& lhs, const uint128& rhs) {
  return (Uint128High64(lhs) == Uint128High64(rhs))
             ? (Uint128Low64(lhs) > Uint128Low64(rhs))
             : (Uint128High64(lhs) > Uint128High64(rhs));
}

inline constexpr bool operator<=(const uint128& lhs, const uint128& rhs) {
  return (Uint128High64(lhs) == Uint128High64(rhs))
             ? (Uint128Low64(lhs) <= Uint128Low64(rhs))
             : (Uint128High64(lhs) <= Uint128High64(rhs));
}

inline constexpr bool operator>=(const uint128& lhs, const uint128& rhs) {
  return (Uint128High64(lhs) == Uint128High64(rhs))
             ? (Uint128Low64(lhs) >= Uint128Low64(rhs))
             : (Uint128High64(lhs) >= Uint128High64(rhs));
}

// Unary operators.

inline constexpr uint128 operator-(const uint128& val) {
  const uint64_t hi_flip = ~Uint128High64(val);
  const uint64_t lo_flip = ~Uint128Low64(val);
  const uint64_t lo_add = lo_flip + 1;
  if (lo_add < lo_flip) {
    return MakeUint128(hi_flip + 1, lo_add);
  }
  return MakeUint128(hi_flip, lo_add);
}

inline constexpr bool operator!(const uint128& val) {
  return !Uint128High64(val) && !Uint128Low64(val);
}

// Logical operators.

inline constexpr uint128 operator~(const uint128& val) {
  return MakeUint128(~Uint128High64(val), ~Uint128Low64(val));
}

inline constexpr uint128 operator|(const uint128& lhs, const uint128& rhs) {
  return MakeUint128(Uint128High64(lhs) | Uint128High64(rhs),
                           Uint128Low64(lhs) | Uint128Low64(rhs));
}

inline constexpr uint128 operator&(const uint128& lhs, const uint128& rhs) {
  return MakeUint128(Uint128High64(lhs) & Uint128High64(rhs),
                           Uint128Low64(lhs) & Uint128Low64(rhs));
}

inline constexpr uint128 operator^(const uint128& lhs, const uint128& rhs) {
  return MakeUint128(Uint128High64(lhs) ^ Uint128High64(rhs),
                           Uint128Low64(lhs) ^ Uint128Low64(rhs));
}

inline constexpr uint128& uint128::operator|=(const uint128& other) {
  hi_ |= other.hi_;
  lo_ |= other.lo_;
  return *this;
}

inline constexpr uint128& uint128::operator&=(const uint128& other) {
  hi_ &= other.hi_;
  lo_ &= other.lo_;
  return *this;
}

inline constexpr uint128& uint128::operator^=(const uint128& other) {
  hi_ ^= other.hi_;
  lo_ ^= other.lo_;
  return *this;
}

// Shift and arithmetic assign operators.

inline constexpr uint128& uint128::operator<<=(int amount) {
  // Shifts of >= 128 are undefined.
  assert(amount < 128);

  // uint64_t shifts of >= 64 are undefined, so we will need some
  // special-casing.
  if (amount < 64) {
    if (amount != 0) {
      hi_ = (hi_ << amount) | (lo_ >> (64 - amount));
      lo_ = lo_ << amount;
    }
  } else {
    hi_ = lo_ << (amount - 64);
    lo_ = 0;
  }
  return *this;
}

inline constexpr uint128& uint128::operator>>=(int amount) {
  // Shifts of >= 128 are undefined.
  assert(amount < 128);

  // uint64_t shifts of >= 64 are undefined, so we will need some
  // special-casing.
  if (amount < 64) {
    if (amount != 0) {
      lo_ = (lo_ >> amount) | (hi_ << (64 - amount));
      hi_ = hi_ >> amount;
    }
  } else {
    lo_ = hi_ >> (amount - 64);
    hi_ = 0;
  }
  return *this;
}

inline constexpr uint128& uint128::operator+=(const uint128& other) {
  hi_ += other.hi_;
  uint64_t lolo = lo_ + other.lo_;
  if (lolo < lo_)
    ++hi_;
  lo_ = lolo;
  return *this;
}

inline constexpr uint128& uint128::operator-=(const uint128& other) {
  hi_ -= other.hi_;
  if (other.lo_ > lo_) --hi_;
  lo_ -= other.lo_;
  return *this;
}

inline constexpr uint128& uint128::operator*=(const uint128& other) {
#if defined(ABSL_HAVE_INTRINSIC_INT128)
  // TODO(strel) Remove once alignment issues are resolved and unsigned __int128
  // can be used for uint128 storage.
  *this = static_cast<unsigned __int128>(*this) *
          static_cast<unsigned __int128>(other);
  return *this;
#else   // ABSL_HAVE_INTRINSIC128
  uint64_t a96 = hi_ >> 32;
  uint64_t a64 = hi_ & 0xffffffff;
  uint64_t a32 = lo_ >> 32;
  uint64_t a00 = lo_ & 0xffffffff;
  uint64_t b96 = other.hi_ >> 32;
  uint64_t b64 = other.hi_ & 0xffffffff;
  uint64_t b32 = other.lo_ >> 32;
  uint64_t b00 = other.lo_ & 0xffffffff;
  // multiply [a96 .. a00] x [b96 .. b00]
  // terms higher than c96 disappear off the high side
  // terms c96 and c64 are safe to ignore carry bit
  uint64_t c96 = a96 * b00 + a64 * b32 + a32 * b64 + a00 * b96;
  uint64_t c64 = a64 * b00 + a32 * b32 + a00 * b64;
  this->hi_ = (c96 << 32) + c64;
  this->lo_ = 0;
  // add terms after this one at a time to capture carry
  *this += uint128(a32 * b00) << 32;
  *this += uint128(a00 * b32) << 32;
  *this += a00 * b00;
  return *this;
#endif  // ABSL_HAVE_INTRINSIC128
}

// Increment/decrement operators.

inline constexpr uint128 uint128::operator++(int) {
  uint128 tmp(*this);
  *this += 1;
  return tmp;
}

inline constexpr uint128 uint128::operator--(int) {
  uint128 tmp(*this);
  *this -= 1;
  return tmp;
}

inline constexpr uint128& uint128::operator++() {
  *this += 1;
  return *this;
}

inline constexpr uint128& uint128::operator--() {
  *this -= 1;
  return *this;
}

namespace {

	// Returns the 0-based position of the last set bit (i.e., most significant bit)
	// in the given uint64_t. The argument may not be 0.
	//
	// For example:
	//   Given: 5 (decimal) == 101 (binary)
	//   Returns: 2
#define STEP(T, n, pos, sh)                   \
  do {                                        \
    if ((n) >= (static_cast<T>(1) << (sh))) { \
      (n) = (n) >> (sh);                      \
      (pos) |= (sh);                          \
    }                                         \
  } while (0)
	static constexpr inline int Fls64(uint64_t n) {
		assert(n != 0);
		int pos = 0;
		STEP(uint64_t, n, pos, 0x20);
		uint32_t n32 = static_cast<uint32_t>(n);
		STEP(uint32_t, n32, pos, 0x10);
		STEP(uint32_t, n32, pos, 0x08);
		STEP(uint32_t, n32, pos, 0x04);
		return pos + ((uint64_t{ 0x3333333322221100 } >> (n32 << 2)) & 0x3);
	}
#undef STEP

	// Like Fls64() above, but returns the 0-based position of the last set bit
	// (i.e., most significant bit) in the given uint128. The argument may not be 0.
	static constexpr inline int Fls128(uint128 n) {
		if (uint64_t hi = Uint128High64(n)) {
			return Fls64(hi) + 64;
		}
		return Fls64(Uint128Low64(n));
	}

	// Long division/modulo for uint128 implemented using the shift-subtract
	// division algorithm adapted from:
	// http://stackoverflow.com/questions/5386377/division-without-using
	inline constexpr void DivModImpl(uint128 dividend, uint128 divisor, uint128* quotient_ret,
		uint128* remainder_ret) {
		assert(divisor != 0);

		if (divisor > dividend) {
			*quotient_ret = 0;
			*remainder_ret = dividend;
			return;
		}

		if (divisor == dividend) {
			*quotient_ret = 1;
			*remainder_ret = 0;
			return;
		}

		uint128 denominator = divisor;
		uint128 quotient = 0;

		// Left aligns the MSB of the denominator and the dividend.
		const int shift = Fls128(dividend) - Fls128(denominator);
		denominator <<= shift;

		// Uses shift-subtract algorithm to divide dividend by denominator. The
		// remainder will be left in dividend.
		for (int i = 0; i <= shift; ++i) {
			quotient <<= 1;
			if (dividend >= denominator) {
				dividend -= denominator;
				quotient |= 1;
			}
			denominator >>= 1;
		}

		*quotient_ret = quotient;
		*remainder_ret = dividend;
	}

	template <typename T>
	inline std::enable_if_t<std::is_floating_point_v<T>,uint128> Initialize128FromFloat(T v) {
		// Rounding behavior is towards zero, same as for built-in types.

		// Undefined behavior if v is NaN or cannot fit into uint128.
		assert(!std::isnan(v) && v > -1 && v < std::ldexp(static_cast<T>(1), 128));

		if (v >= std::ldexp(static_cast<T>(1), 64)) {
			uint64_t hi = static_cast<uint64_t>(std::ldexp(v, -64));
			uint64_t lo = static_cast<uint64_t>(v - std::ldexp(static_cast<T>(hi), 64));
			return MakeUint128(hi, lo);
		}

		return MakeUint128(0, static_cast<uint64_t>(v));
	}
}  // namespace
//
//uint128::uint128(float v) : uint128(Initialize128FromFloat(v)) {}
//uint128::uint128(double v) : uint128(Initialize128FromFloat(v)) {}
//uint128::uint128(long double v) : uint128(Initialize128FromFloat(v)) {}

inline constexpr uint128& uint128::operator/=(const uint128& divisor) {
	uint128 quotient = 0;
	uint128 remainder = 0;
	DivModImpl(*this, divisor, &quotient, &remainder);
	*this = quotient;
	return *this;
}
inline constexpr uint128& uint128::operator%=(const uint128& divisor) {
	uint128 quotient = 0;
	uint128 remainder = 0;
	DivModImpl(*this, divisor, &quotient, &remainder);
	*this = remainder;
	return *this;
}

inline std::ostream& operator<<(std::ostream& o, const uint128& b) {
	std::ios_base::fmtflags flags = o.flags();

	// Select a divisor which is the largest power of the base < 2^64.
	uint128 div;
	int div_base_log;
	switch (flags & std::ios::basefield) {
	case std::ios::hex:
		div = 0x1000000000000000;  // 16^15
		div_base_log = 15;
		break;
	case std::ios::oct:
		div = 01000000000000000000000;  // 8^21
		div_base_log = 21;
		break;
	default:  // std::ios::dec
		div = 10000000000000000000u;  // 10^19
		div_base_log = 19;
		break;
	}

	// Now piece together the uint128 representation from three chunks of
	// the original value, each less than "div" and therefore representable
	// as a uint64_t.
	std::ostringstream os;
	std::ios_base::fmtflags copy_mask =
		std::ios::basefield | std::ios::showbase | std::ios::uppercase;
	os.setf(flags & copy_mask, copy_mask);
	uint128 high = b;
	uint128 low;
	DivModImpl(high, div, &high, &low);
	uint128 mid;
	DivModImpl(high, div, &high, &mid);
	if (Uint128Low64(high) != 0) {
		os << Uint128Low64(high);
		os << std::noshowbase << std::setfill('0') << std::setw(div_base_log);
		os << Uint128Low64(mid);
		os << std::setw(div_base_log);
	}
	else if (Uint128Low64(mid) != 0) {
		os << Uint128Low64(mid);
		os << std::noshowbase << std::setfill('0') << std::setw(div_base_log);
	}
	os << Uint128Low64(low);
	std::string rep = os.str();

	// Add the requisite padding.
	std::streamsize width = o.width(0);
	if (static_cast<size_t>(width) > rep.size()) {
		if ((flags & std::ios::adjustfield) == std::ios::left) {
			rep.append(width - rep.size(), o.fill());
		}
		else {
			rep.insert(0, width - rep.size(), o.fill());
		}
	}

	// Stream the final representation in a single "<<" call.
	return o << rep;
}



} // namespace pcg_extras

#endif // PCG_UINT128_HPP_INCLUDED
