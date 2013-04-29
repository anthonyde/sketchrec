#include <algorithm>
#include <cassert>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <cairomm/context.h>
#include <gtkmm/actiongroup.h>
#include <gtkmm/aspectframe.h>
#include <gtkmm/box.h>
#include <gtkmm/button.h>
#include <gtkmm/drawingarea.h>
#include <gtkmm/label.h>
#include <gtkmm/main.h>
#include <gtkmm/stock.h>
#include <gtkmm/uimanager.h>
#include <gtkmm/window.h>
#include <sigc++/sigc++.h>

#include "features.h"
#include "io.h"
#include "svm.h"
#include "types.h"

struct point {
  double x, y;
};

typedef std::vector< point > path_type;

// The width of a line as a fraction of the size of the image.
const double line_width = 0.00375;

const int sketch_timeout = 500; // ms
const int sketch_min_size = 256; // px

class SketchArea : public Gtk::DrawingArea
{
public:
  SketchArea() {
    set_size_request(sketch_min_size, sketch_min_size);
    add_events(Gdk::BUTTON_PRESS_MASK | Gdk::EXPOSURE_MASK |
      Gdk::POINTER_MOTION_MASK | Gdk::POINTER_MOTION_HINT_MASK);
  }

  virtual ~SketchArea() {
  }

  // Draw the current sketch to a matrix, converting it to grayscale.
  template< class T, long NR, long NC >
  void draw(dlib::matrix< T, NR, NC > &image) const {
    Cairo::RefPtr< Cairo::ImageSurface > surface =
      Cairo::ImageSurface::create(Cairo::FORMAT_RGB24, NC, NR);
    Cairo::RefPtr< Cairo::Context > cr = Cairo::Context::create(surface);

    cr->set_source_rgb(1.0, 1.0, 1.0);
    cr->paint();

    cr->scale(NC, NR);
    draw(cr);

    cr->show_page();

    {
      const unsigned char *p = surface->get_data();
      const int stride = surface->get_stride();

      const unsigned char *q = p;
      for (long j = 0; j < NR; ++j, q = p += stride) {
        for (long i = 0; i < NC; ++i, q += 4) {
          // Convert the image to grayscale.
          image(j, i) = (.299 * q[0] + .587 * q[1] + .114 * q[2]) / 255.;
        }
      }
    }
  }

  // Scale and center the image to fit the canvas.
  void scale() {
    if (paths.empty())
      return;

    double x0, x1, y0, y1;
    x0 = y0 = 1.0;
    x1 = y1 = 0.0;
    for (const auto &path : paths) {
      for (const point &p : path) {
        x0 = std::min(x0, p.x);
        x1 = std::max(x1, p.x);
        y0 = std::min(y0, p.y);
        y1 = std::max(y1, p.y);
      }
    }

    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double dmax = std::max(dx, dy);

    for (auto &path : paths) {
      for (point &p : path) {
        p.x = 0.5 + (p.x - x0 - dx / 2.0) * 0.8 / dmax;
        p.y = 0.5 + (p.y - y0 - dy / 2.0) * 0.8 / dmax;
      }
    }

    signal_update.emit();
    invalidate();
  }

  // Clear the sketch, removing all paths.
  void clear() {
    paths.clear();
    signal_update.emit();
    invalidate();
  }

  sigc::signal< void > signal_update;

protected:
  // Invalidate the entire drawing area.
  void invalidate() {
    Gtk::Allocation allocation = get_allocation();
    const int width = allocation.get_width();
    const int height = allocation.get_height();
    Gdk::Rectangle rect(0, 0, width, height);
    get_window()->invalidate_rect(rect, false);
  }

  // Draw all paths to a Cairo context.
  void draw(const Cairo::RefPtr< Cairo::Context > &cr) const {
    cr->set_source_rgb(0.0, 0.0, 0.0);
    cr->set_line_width(line_width);
    cr->set_line_cap(Cairo::LINE_CAP_ROUND);
    cr->set_line_join(Cairo::LINE_JOIN_ROUND);

    for (const auto &path : paths) {
      if (path.empty())
        continue;

      path_type::const_iterator it = path.begin();
      cr->move_to(it->x, it->y);
      for (; it < path.end(); ++it)
        cr->line_to(it->x, it->y);
      cr->stroke();
    }
  }

  virtual bool on_button_press_event(GdkEventButton *event) {
    if (event->button == 1) {
      // Create a new path starting at the current cursor position.
      if (paths.empty() || !get_path().empty()) {
        paths.push_back(path_type());
        add_point(event->x, event->y);
      }
    }

    return true;
  }

  virtual bool on_expose_event(GdkEventExpose *) {
    Cairo::RefPtr< Cairo::Context > cr = get_window()->create_cairo_context();

    Gtk::Allocation allocation = get_allocation();
    const int width = allocation.get_width();
    const int height = allocation.get_height();

    cr->set_source_rgb(1.0, 1.0, 1.0);
    cr->paint();

    cr->scale(width, height);
    draw(cr);

    cr->show_page();

    return true;
  }

  virtual bool on_motion_notify_event(GdkEventMotion *event) {
    int x, y;
    Gdk::ModifierType state;

    if (event->is_hint) {
      get_window()->get_pointer(x, y, state);
    }
    else {
      x = static_cast< int >(event->x);
      y = static_cast< int >(event->y);
      state = static_cast< Gdk::ModifierType >(event->state);
    }

    if (!paths.empty() && state & Gdk::BUTTON1_MASK) {
      // Add a point to the current path if the first button is pressed.
      add_point(x, y);
    }

    return true;
  }

  // Return the current path.
  path_type &get_path() {
    assert(!paths.empty());
    return paths.back();
  }

  // Add a point (in widget coordinates) to the current path.
  void add_point(int x, int y) {
    Gtk::Allocation allocation = get_allocation();
    const int width = allocation.get_width();
    const int height = allocation.get_height();

    point p;
    double x0, x1, y0, y1;
    p.x = x0 = x1 = static_cast< double >(x) / width;
    p.y = y0 = y1 = static_cast< double >(y) / height;

    path_type &path = get_path();
    if (!path.empty()) {
      const point &pp = path.back();
      x0 = std::min(x0, pp.x);
      x1 = std::max(x1, pp.x);
      y0 = std::min(y0, pp.y);
      y1 = std::max(y1, pp.y);
    }

    path.push_back(p);

    Gdk::Rectangle rect(std::floor((x0 - line_width) * width),
      std::floor((y0 - line_width) * height),
      std::ceil((x1 - x0 + 2 * line_width) * width),
      std::ceil((y1 - y0 + 2 * line_width) * height));

    signal_update.emit();
    get_window()->invalidate_rect(rect, false);
  }

  std::vector< path_type > paths;
};

class MainWindow : public Gtk::Window
{
public:
  MainWindow(const vocab_type *vocab_, const std::map< int, std::string >
    *cat_map_, bool ova_, df_type *df_) : hbox(true, 10),
    vocab(vocab_), cat_map(cat_map_), ova(ova_), df(df_) {
    // Set up the window.
    set_title("Sketch recognition");
    set_size_request(800, 400);

    add(vbox);

    Glib::RefPtr< Gtk::ActionGroup > action_group =
      Gtk::ActionGroup::create();

    action_group->add(Gtk::Action::create("New", Gtk::Stock::NEW, "_New",
      "Create a new sketch"), sigc::mem_fun(*this, &MainWindow::on_new));
    action_group->add(Gtk::Action::create("ScaleToFit", Gtk::Stock::ZOOM_FIT,
      "_Scale to Fit", "Scale the current sketch to fit the canvas"),
      sigc::mem_fun(*this, &MainWindow::on_scale));
    action_group->add(Gtk::Action::create("Quit", Gtk::Stock::QUIT, "_Quit"),
      sigc::mem_fun(*this, &MainWindow::on_quit));

    Glib::RefPtr< Gtk::UIManager > ui_manager = Gtk::UIManager::create();
    ui_manager->insert_action_group(action_group);
    add_accel_group(ui_manager->get_accel_group());

    Glib::ustring ui_info =
      "<ui>"
      "  <toolbar name=\"ToolBar\">"
      "    <toolitem action=\"New\" />"
      "    <toolitem action=\"ScaleToFit\" />"
      "    <toolitem action=\"Quit\" />"
      "  </toolbar>"
      "</ui>";
    ui_manager->add_ui_from_string(ui_info);

    vbox.pack_start(*ui_manager->get_widget("/ToolBar"), Gtk::PACK_SHRINK);

    hbox.set_border_width(10);
    vbox.pack_start(hbox);

    sketch_frame.set_shadow_type(Gtk::SHADOW_NONE);
    hbox.pack_start(sketch_frame);

    sketch.signal_update.connect(sigc::mem_fun(*this,
      &MainWindow::on_sketch_update));
    sketch_frame.add(sketch);

    cat_label.set_text("Draw in the box to begin.");
    cat_label.set_justify(Gtk::JUSTIFY_CENTER);
    cat_label.set_line_wrap();
    hbox.pack_start(cat_label);

    show_all();
  }

  virtual ~MainWindow() {
  }

protected:
  virtual void on_new() {
    sketch.clear();
  }

  virtual void on_scale() {
    sketch.scale();
  }

  virtual void on_quit() {
    hide();
  }

  virtual void on_sketch_update() {
    if (sketch_timer_conn.connected())
      sketch_timer_conn.disconnect();

    sketch_timer_conn = Glib::signal_timeout().connect(sigc::mem_fun(*this,
      &MainWindow::on_sketch_timeout), sketch_timeout);
  }

  virtual bool on_sketch_timeout() {
    image_type image;
    sketch.draw(image);
    image = 1. - image;

    std::vector< feature_desc_type > descs;
    extract_descriptors(image, descs);

    feature_hist_type hist;
    feature_hist(descs, *vocab, hist);

    const int cat = ova ? df->get< ova_df_type >()(hist) :
      df->get< ovo_df_type >()(hist);

    const auto it = cat_map->find(cat);
    assert(it != cat_map->end());

    std::ostringstream ss;
    ss << "<span size=\"xx-large\">" << it->second << "</span>";
    cat_label.set_markup(ss.str());

    return false;
  }

  sigc::connection sketch_timer_conn;

  Gtk::VBox vbox;
  Gtk::HBox hbox;
  Gtk::AspectFrame sketch_frame;
  SketchArea sketch;
  Gtk::Label cat_label;

  const vocab_type *vocab;
  const std::map< int, std::string > *cat_map;
  bool ova;
  df_type *df;
};

int main(int argc, char* argv[])
{
  Gtk::Main app(argc, argv);

  // Process the command-line arguments.
  const char *vocab_path = "vocab.out";
  const char *map_path = "map_id_label.txt";
  const char *cats_path = "cats.out";
  bool ova = true;

  {
    int i;
    for (i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "-h")) {
        goto usage;
      }
      else if (!strcmp(argv[i], "-v")) {
        vocab_path = argv[++i];
      }
      else if (!strcmp(argv[i], "-m")) {
        map_path = argv[++i];
      }
      else if (!strcmp(argv[i], "-c")) {
        ++i;
        if (!strcmp(argv[i], "ova")) {
          ova = true;
        }
        else if (!strcmp(argv[i], "ovo")) {
          ova = false;
        }
        else {
          std::cerr << argv[0] << ": Unsupported classifier: `" << argv[i]
            << "'\n";
          goto err;
        }
      }
      else {
        break;
      }
    }

    if (i < argc)
      cats_path = argv[i++];

    if (i != argc)
      goto usage;

    if (!vocab_path || !map_path || !cats_path)
      goto usage;
  }

  {
    // Load the vocabulary.
    vocab_type vocab;
    {
      std::ifstream fs(vocab_path, std::ios::binary);
      deserialize2(vocab, fs);
    }

    // Load the category map.
    std::map< int, std::string > cat_map;
    {
      std::ifstream fs(map_path);
      for (std::string line; std::getline(fs, line);) {
        std::istringstream ss(line);
        int i;
        std::string label;
        ss >> i;
        ss.get(); // ','
        std::getline(ss, label);
        cat_map[i] = label;
      }
    }

    // Load the category classifier.
    df_type df;
    {
      std::ifstream fs(cats_path, std::ios::binary);
      if (ova)
        deserialize2(df.get< ova_df_type >(), fs);
      else
        deserialize2(df.get< ovo_df_type >(), fs);
    }

    MainWindow win(&vocab, &cat_map, ova, &df);
    app.run(win);
  }

  return 0;

usage:
  std::cerr << "Usage: " << argv[0]
    << " [-v vocab-file] [-m map-file] [-c classifier] [cats-file]\n";

err:
  return 1;
}
