#include <glib-object.h>

// A static class for automatically initializing GLib
struct glib_initializer {
  glib_initializer() {
    g_type_init();
  }
};

const glib_initializer glib_init;

