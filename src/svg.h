#ifndef SVG_H
#define SVG_H

#include <memory>

#include <cairo.h>
#include <dlib/matrix.h>
#include <glib-object.h>
#include <librsvg/rsvg.h>

// An error while loading an image
struct image_error : std::exception {
  virtual ~image_error() noexcept {
  }

  virtual const char *what() const noexcept {
    return "image error";
  }
};

// Destruction policies for Cairo objects
// Ideally these would be wrapped in classes, but they aren't used enough.
template< class T > struct cairo_delete {};

template<> struct cairo_delete< cairo_t > {
  void operator()(cairo_t *p) {
    cairo_destroy(p);
  }
};

template<> struct cairo_delete< cairo_surface_t > {
  void operator()(cairo_surface_t *p) {
    cairo_surface_destroy(p);
  }
};

// A destruction policy for GLib objects
template< class T > struct glib_delete {
  void operator()(gpointer p) {
    g_object_unref(p);
  }
};

// Load an SVG file, storing the rasterized image in a square matrix.
template< class T, long N >
void load_svg(const char *file, dlib::matrix< T, N, N > &image) {
  std::unique_ptr< cairo_surface_t, cairo_delete< cairo_surface_t > >
    surface(cairo_image_surface_create(CAIRO_FORMAT_RGB24, N, N));
  if (!surface)
    throw image_error();

  std::unique_ptr< cairo_t, cairo_delete< cairo_t > >
    cr(cairo_create(surface.get()));
  if (cairo_status(cr.get()) != CAIRO_STATUS_SUCCESS)
    throw image_error();

  // Clear the buffer to white.
  cairo_set_source_rgb(cr.get(), 1., 1., 1.);
  cairo_paint(cr.get());

  {
    GError *error;
    std::unique_ptr< RsvgHandle, glib_delete< RsvgHandle > >
      svg(rsvg_handle_new_from_file(file, &error));
    if (!svg)
      throw image_error();

    RsvgDimensionData dims;
    rsvg_handle_get_dimensions(svg.get(), &dims);

    // Loaded images must be square.
    if (dims.width != dims.height)
      throw image_error();

    const double scale = static_cast< double >(N) / dims.width;
    cairo_scale(cr.get(), scale, scale);

    gboolean res;
    #pragma omp critical
    {
      res = rsvg_handle_render_cairo(svg.get(), cr.get());
    }
    if (!res)
      throw image_error();
  }

  cairo_surface_flush(surface.get());

  // Store the image in the matrix.
  {
    const unsigned char *p = cairo_image_surface_get_data(surface.get());
    const int stride = cairo_image_surface_get_stride(surface.get());

    const unsigned char *q = p;
    for (long j = 0; j < N; ++j, q = p += stride) {
      for (long i = 0; i < N; ++i, q += 4) {
        // Convert the image to grayscale.
        image(j, i) = (.299 * q[0] + .587 * q[1] + .114 * q[2]) / 255.;
      }
    }
  }
}

#endif
