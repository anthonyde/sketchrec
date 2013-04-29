noinst_PROGRAMS = cats classify gui vocab

AM_CXXFLAGS = $(CAIRO_CFLAGS) $(FFTW_CFLAGS) $(GLIB_CFLAGS) $(GTKMM_CFLAGS) $(LIBRSVG_CFLAGS) $(OPENMP_CXXFLAGS)
AM_LDFLAGS = $(CAIRO_LIBS) $(FFTW_LIBS) $(GLIB_LIBS) $(GTKMM_LIBS) $(LIBRSVG_LIBS)

cats_SOURCES = cats.cpp svg.cpp util.cpp
classify_SOURCES = classify.cpp svg.cpp util.cpp
gui_SOURCES = gui.cpp util.cpp
vocab_SOURCES = vocab.cpp svg.cpp util.cpp
