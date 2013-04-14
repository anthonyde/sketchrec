# Human Sketch Recognition

This project implements a human sketch recognition algorithm based on the
paper:

Mathias Eitz, James Hays, and Marc Alexa. *[How Do Humans Sketch Objects?]
[1]* ACM Trans. Graph. (Proc. SIGGRAPH), 31(4):44:1-10, July 2012.

[1]: http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/

The sketch dataset is not stored in this repository and must be downloaded
separately.  See the setup instructions below for details.

## Setup

### Dependencies

GCC 4.6.3 or newer with support for [OpenMP] [2], autoconf 2.69, and automake
1.12 are required to compile the code.

[2]: http://openmp.org/

The following libraries must also be installed before compiling:

  * [cairo] [3] (>= 1.10)

  * [dlib] [4] (>= 18.0)

  * [fftw] [5] (>= 3.0, configured with single precision support)

  * [librsvg] [6] (>= 2.0)

[3]: http://cairographics.org/
[4]: http://dlib.net/
[5]: http://fftw.org/
[6]: https://live.gnome.org/LibRsvg

### Compiling

In the root directory of the project, run `autogen.sh` to set up the
configuration scripts.

    $ ./autogen.sh

Create the build directory and switch to it, then run `configure` and `make`
to compile.

    $ mkdir build
    $ cd build
    $ ../configure
    $ make

To compile without optimizations (for debugging), configure with:

    $ ../configure CXXFLAGS='-O0'

The compiled programs in the build directory mirror the source directory
structure.

### Data

Run `util/get-data` from the root directory to automatically download the
sketch dataset into the data directory.  The dataset can also be downloaded
manually from the following link:

[Sketch dataset (SVG)] [7] (zip, ~50 MB)

[7]: http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip

## Running

The easiest way to train a classifier and classify sketch data is to run the
provided utility scripts from the project root directory.  These scripts make
some assumptions about the project layout:

  * The build directory is in the project root and is called `build`

  * The SVG dataset is in `data/svg` and has one subdirectory for each
    category

  * Each category subdirectory contains only sketch images and has no nested
    subdirectories

Each script accepts the same command-line arguments as the associated program
and runs with the same default arguments (with one exception: default paths
are prefixed with `data/`).  For detailed information about these arguments,
see **Programs**.

To generate the visual vocabulary for the entire sketch dataset, run:

    $ util/run-vocab [-n sample-count]

By default, this script runs with 1,000,000 features selected at random from
the dataset.  This usually takes between 60 and 90 minutes (400-600 iterations
of k-means clustering) to complete on a 2.2 GHz Core i7 with 8 threads.

The dataset is organized into 8 folds to aid in selecting subsets of the data.
Each fold is assigned an index (0-7), and folds can be negated by prepending
a ~.  The identifier ~4, for example, refers to the contents of folds 0-3 and
5-7.

To train a classifier on a subset of the data, run:

    $ util/run-cats [--fold fold-id] [-c classifier] [cats-file]

By default, this script operates on folds 1-7 (~0) and trains a one-vs-all
classifier.  Training a one-vs-one classifier on this dataset requires nearly
8 GB of memory.

To classify a subset of the data, run:

    $ util/run-classify [--fold fold-id] [-c classifier] [cats-file]

By default, this script operates on fold 0 and expects a one-vs-all
classifier.

### Demo

The default settings will create a working classifier trained on 7/8 (87.5%)
of the dataset.  To train, after compiling and obtaining the dataset, run:

    $ util/run-vocab
    $ util/run-cats

This will create the files `data/vocab.out`, the visual vocabulary, and
`data/cats.out`, the one-vs-all classifier.

To classify the un-trained portion of the dataset, run:

    $ util/run-classify

### Programs

These programs can be found under the build directory after running `make`.
Arguments in brackets are optional and will assume default values when
omitted.

  * `vocab [-n sample-count] [vocab-file]`

    Generate a visual vocabulary for the images specified on standard input,
    one path per line.  Feature descriptors are extracted from each file.
    `sample-count` (default: 1,000,000) random descriptors are selected from
    this dataset and clustered into 500 visual words.  The resulting
    vocabulary is written to `vocab-file` (default: `vocab.out`).

  * `cats [-v vocab-file] [-m map-file] [-c classifier] [cats-file]`

    Train a classifier with the images specified on standard input, one path
    per line.  The name of the subdirectory containing each image is used as
    the category label.  Feature histograms are generated from each image for
    training using `vocab-file` (default: `vocab.out`).  The mapping between
    category labels and numeric identifiers is read from `map-file` (default:
    `map_id_label.txt`).  Two types of classifiers are currently supported,
    one-vs-all (`ova`) and one-vs-one (`ovo`).  `classifier` (default: `ova`)
    must be one of these two values.  The resulting classifier is written to
    `cats-file` (default: `cats.out`).

  * `classify [-v vocab-file] [-m map-file] [-c classifier] [cats-file]`

    Run a classifier on each image specified on standard input, one path per
    line.  Each path and its predicted category is written to standard output.
    The default values for each argument are the same as above.  The same
    classifier type must be selected for both training and classification,
    since this information is currently not stored with the classifier.

## License

The files in this project are released under the BSD-3 license unless stated
otherwise.  See the file `LICENSE` for details.
