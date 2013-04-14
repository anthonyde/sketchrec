#include <dlib/matrix.h>

#include "util.h"

dlib::matrix< float, 3, 3 > sobel_x_init() {
  dlib::matrix< float, 3, 3 > m;
  m =
    -1.f,  0.f,  1.f,
    -2.f,  0.f,  2.f,
    -1.f,  0.f,  1.f;
  return m;
}

const dlib::matrix< float, 3, 3 > sobel_x = sobel_x_init();

dlib::matrix< float, 3, 3 > sobel_y_init() {
  dlib::matrix< float, 3, 3 > m;
  m =
    -1.f, -2.f, -1.f,
     0.f,  0.f,  0.f,
     1.f,  2.f,  1.f;
  return m;
}

const dlib::matrix< float, 3, 3 > sobel_y = sobel_y_init();

