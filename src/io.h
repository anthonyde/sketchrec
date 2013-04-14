#ifndef IO_H
#define IO_H

#include <cassert>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <dlib/unordered_pair.h>

// An error while serializing or deserializing
struct serialization_error : std::exception {
  virtual ~serialization_error() noexcept {
  }

  virtual const char *what() const noexcept {
    return "serialization error";
  }
};

// Check whether a type is a multi-class decision function.
template< class T >
struct is_multiclass_df : std::false_type {};

template< class T, class DF1 >
struct is_multiclass_df< dlib::one_vs_all_decision_function< T, DF1 > >
  : std::true_type {};

template< class T, class DF1 >
struct is_multiclass_df< dlib::one_vs_one_decision_function< T, DF1 > >
  : std::true_type {};

// Template prototypes

template< class T1, class T2 >
void serialize2(const std::pair< T1, T2 > &x, std::ostream &s);

template< class T1, class T2 >
void deserialize2(std::pair< T1, T2 > &x, std::istream &s);

template< class CharT >
void serialize2(const std::basic_string< CharT > &x, std::ostream &s);

template< class CharT >
void deserialize2(std::basic_string< CharT > &x, std::istream &s);

template< class T >
void serialize2(const std::vector< T > &xs, std::ostream &s);

template< class T >
void deserialize2(std::vector< T > &xs, std::istream &s);

template< class Key, class T >
void serialize2(const std::multimap< Key, T > &xs, std::ostream &s);

template< class Key, class T >
void deserialize2(std::multimap< Key, T > &xs, std::istream &s);

template< class T, long NR, long NC >
void serialize2(const dlib::matrix< T, NR, NC > &x, std::ostream &s);

template< class T, long NR, long NC >
void deserialize2(dlib::matrix< T, NR, NC > &x, std::istream &s);

template< class T >
void serialize2(const dlib::unordered_pair< T > &xs, std::ostream &s);

template< class T >
void deserialize2(dlib::unordered_pair< T > &xs, std::istream &s);

template< class T >
void serialize2(const dlib::radial_basis_kernel< T > &x, std::ostream &s);

template< class T >
void deserialize2(dlib::radial_basis_kernel< T > &x, std::istream &s);

template< class K >
void serialize2(const dlib::decision_function< K > &x, std::ostream &s);

template< class K >
void deserialize2(dlib::decision_function< K > &x, std::istream &s);

template< template< class... > class DF, class T, class DF1, class... DFS >
typename std::enable_if< is_multiclass_df< DF< T, DF1, DFS... > >::value >::type
  serialize2(const DF< T, DF1, DFS... > &x, std::ostream &s);

template< template< class... > class DF, class T, class DF1, class... DFS >
typename std::enable_if< is_multiclass_df< DF< T, DF1, DFS... > >::value >::type
  deserialize2(DF< T, DF1, DFS... > &x, std::istream &s);

// Arithmetic types

template< class T >
typename std::enable_if< std::is_arithmetic< T >::value >::type
serialize2(const T &x, std::ostream &s) {
  assert(!std::is_floating_point< T >::value || !std::isnan(x));
  if (!s.write(reinterpret_cast< const char * >(&x), sizeof(T)))
    throw serialization_error();
}

template< class T >
typename std::enable_if< std::is_arithmetic< T >::value >::type
deserialize2(T &x, std::istream &s) {
  if (!s.read(reinterpret_cast< char * >(&x), sizeof(T)))
    throw serialization_error();
  assert(!std::is_floating_point< T >::value || !std::isnan(x));
}

// std::pair

template< class T1, class T2 >
void serialize2(const std::pair< T1, T2 > &x, std::ostream &s) {
  serialize2(x.first, s);
  serialize2(x.second, s);
}

template< class T1, class T2 >
void deserialize2(std::pair< T1, T2 > &x, std::istream &s) {
  deserialize2(x.first, s);
  deserialize2(x.second, s);
}

// std::basic_string

template< class CharT >
void serialize2(const std::basic_string< CharT > &x, std::ostream &s) {
  const auto size = x.size();
  serialize2(size, s);
  if (!s.write(&x[0], sizeof(CharT) * size))
    throw serialization_error();
}

template< class CharT >
void deserialize2(std::basic_string< CharT > &x, std::istream &s) {
  typename std::basic_string< CharT >::size_type size;
  std::vector< CharT > data;
  deserialize2(size, s);
  data.resize(size);
  if (!s.read(&data[0], sizeof(CharT) * size))
    throw serialization_error();
  x = std::basic_string< CharT >(data.begin(), data.end());
}

// std::vector

template< class T >
void serialize2(const std::vector< T > &xs, std::ostream &s) {
  const auto size = xs.size();
  serialize2(size, s);
  for (const auto &x : xs)
    serialize2(x, s);
}

template< class T >
void deserialize2(std::vector< T > &xs, std::istream &s) {
  typename std::vector< T >::size_type size;
  deserialize2(size, s);
  xs.resize(size);
  for (auto &x : xs)
    deserialize2(x, s);
}

// std::multimap

template< class Key, class T >
void serialize2(const std::multimap< Key, T > &xs, std::ostream &s) {
  const auto size = xs.size();
  serialize2(size, s);
  for (auto &pair : xs)
    serialize2(pair, s);
}

template< class Key, class T >
void deserialize2(std::multimap< Key, T > &xs, std::istream &s) {
  typename std::multimap< Key, T >::size_type size;
  deserialize2(size, s);
  for (typename std::multimap< Key, T >::size_type i = 0; i < size; ++i) {
    std::pair< Key, T > pair;
    deserialize2(pair, s);
    xs.insert(pair);
  }
}

// dlib::matrix

template< class T, long NR, long NC >
void serialize2(const dlib::matrix< T, NR, NC > &x, std::ostream &s) {
  const long rows = x.nr();
  const long cols = x.nc();
  serialize2(rows, s);
  serialize2(cols, s);
  for (long j = 0; j < rows; ++j) {
    for (long i = 0; i < cols; ++i)
      serialize2(x(j, i), s);
  }
}

template< class T, long NR, long NC >
void deserialize2(dlib::matrix< T, NR, NC > &x, std::istream &s) {
  long rows, cols;
  deserialize2(rows, s);
  deserialize2(cols, s);
  x.set_size(rows, cols);
  for (long j = 0; j < rows; ++j) {
    for (long i = 0; i < cols; ++i)
      deserialize2(x(j, i), s);
  }
}

// dlib::unordered_pair

template< class T >
void serialize2(const dlib::unordered_pair< T > &xs, std::ostream &s) {
  serialize2(xs.first, s);
  serialize2(xs.second, s);
}

template< class T >
void deserialize2(dlib::unordered_pair< T > &xs, std::istream &s) {
  deserialize2(const_cast< T & >(xs.first), s);
  deserialize2(const_cast< T & >(xs.second), s);
}

// dlib::radial_basis_kernel

template< class T >
void serialize2(const dlib::radial_basis_kernel< T > &x, std::ostream &s) {
  serialize2(x.gamma, s);
}

template< class T >
void deserialize2(dlib::radial_basis_kernel< T > &x, std::istream &s) {
  typedef dlib::radial_basis_kernel< T > kernel_type;
  deserialize2(const_cast< typename kernel_type::scalar_type & >(x.gamma), s);
}

// dlib::decision_function

template< class K >
void serialize2(const dlib::decision_function< K > &x, std::ostream &s) {
  serialize2(x.alpha, s);
  serialize2(x.b, s);
  serialize2(x.kernel_function, s);
  serialize2(x.basis_vectors, s);
}

template< class K >
void deserialize2(dlib::decision_function< K > &x, std::istream &s) {
  deserialize2(x.alpha, s);
  deserialize2(x.b, s);
  deserialize2(x.kernel_function, s);
  deserialize2(x.basis_vectors, s);
}

// dlib::one_vs_all_decision_function
// dlib::one_vs_one_decision_function

template< template< class... > class DF, class T, class DF1, class... DFS >
typename std::enable_if< is_multiclass_df< DF< T, DF1, DFS... > >::value >::type
  serialize2(const DF< T, DF1, DFS... > &x, std::ostream &s) {
  const auto &dfs = x.get_binary_decision_functions();
  const auto size = dfs.size();
  serialize2(size, s);
  for (const auto &pair : dfs) {
    serialize2(std::make_pair(pair.first,
      dlib::any_cast< DF1 >(pair.second)), s);
  }
}

template< template< class... > class DF, class T, class DF1, class... DFS >
typename std::enable_if< is_multiclass_df< DF< T, DF1, DFS... > >::value >::type
  deserialize2(DF< T, DF1, DFS... > &x, std::istream &s) {
  typedef DF< T, DF1, DFS... > df_type;
  typedef typename df_type::binary_function_table binary_function_table;
  typename binary_function_table::size_type size;
  binary_function_table dfs;
  deserialize2(size, s);
  for (typename binary_function_table::size_type i = 0; i < size; ++i) {
    std::pair< typename binary_function_table::key_type, DF1 > pair;
    deserialize2(pair, s);
    dfs.insert(pair);
  }
  x = df_type(dfs);
}

#endif
