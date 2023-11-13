#pragma once

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <queue>
#include <variant>
#include <vector>
#include <string>
#include <optional>
#include <stdarg.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <map>

#include "glm/glm.hpp"


// This is included deliberately:
#include <cmath>

// This is supposed to be a DSL, so let's define a few useful shorcuts from glm:

using vec2 = glm::dvec2;
using vec3 = glm::dvec3;


struct range { double start, end; };

template <>
struct std::hash<range> {
    size_t operator()(const range &span) {
        std::hash<double> double_hasher;
        size_t hash = double_hasher(span.start) ^ double_hasher(span.end);
        return hash;
    }
};


enum interpolation_mode { SMOOTH, LINE, NONE };

struct plot {
    std::vector<vec2> points;
    std::optional<std::string> color;

    double mark_size = 1.0;
    interpolation_mode interpolate = LINE;

    std::optional<std::string> name; // TODO: move name up


    // TODO: This should become private, or not exist (but I don't want to lose aggregate status)
    //       Cringe... but it's already so late, I'll think about it later!
    mutable bool is_shown = false;


    template <typename function_type>
    static plot analytical(function_type &&f, range range = {-1, 1},
                           int initial_points = 100, double max_delta = 0.0003) {
        // TODO: all this should be calculated in relative coordinates!

        struct segment { vec2 beg, end; };
        std::queue<segment> staged_segments;

        // Samples of the function:
        std::vector<vec2> points;

        vec2 previous_sample = { 0, 0 };
        double step = (range.end - range.start) / initial_points;
        for (double x = range.start; x <= range.end; x += step) {
            double y = f(x);
            vec2 current_sample = { x, y };

            points.push_back(current_sample);

            if (x != range.start)
                staged_segments.push({ previous_sample, current_sample });

            previous_sample = current_sample;
        }

        while (!staged_segments.empty()) {
            segment segment = staged_segments.front();
            staged_segments.pop();

            vec2 mid = (segment.beg + segment.end) / 2.0;
            vec2 sample = { mid.x, f(mid.x) };

            double length = glm::length(mid - sample);

            // Add another sample:
            if (length > max_delta) {
                points.push_back({ sample.x, sample.y });

                // Now there's two segments:
                staged_segments.push({ segment.beg, sample });
                staged_segments.push({ segment.end, sample });
            }
        }

        std::sort(points.begin(), points.end(), [](const auto &lhs, const auto &rhs) { return lhs.x < rhs.x; });

        return { std::move(points), {}, 0 };
    }



    template <typename function_type>
    static plot analytical_(function_type &&f, range range = { -1, 1 }, int generated_points = 1500) {
        std::vector<vec2> points;
        points.reserve(generated_points);

        double step = (range.end - range.start) / generated_points;
        for (double x = range.start; x <= range.end; x += step) {
            double y = f(x);
            vec2 new_point { x, y };

            points.push_back(new_point);
        }

        return { std::move(points), {}, 0 };
    }

    ~plot();
};


template <>
struct std::hash<plot> {
    size_t operator()(const plot &current) {
        size_t hash = 0;

        std::hash<double> double_hasher;
        for (const auto &point: current.points)
            hash ^= double_hasher(point.x) ^ double_hasher(point.y);

        hash ^= double_hasher(current.mark_size);
        hash ^= std::hash<std::optional<std::string>>{}(current.color);

        hash ^= current.interpolate;
        return hash;
    }
};


enum axis_mode { LINEAR, LOG };


struct axis {
    std::optional<std::string> label;
    axis_mode mode = LINEAR;

    std::optional<range> span;

    static range calculate_span(const std::vector<vec2> &points, double vec2::*coordinate) {
        assert(!points.empty());

        auto [min, max] = std::minmax_element(points.begin(), points.end(),
            [&](auto &&lhs, auto &&rhs) {
                return lhs.*coordinate < rhs.*coordinate;
            }
        );

        return { (*min).*coordinate, (*max).*coordinate };
    }

    // TODO: support tick configuration:
    //
    // struct tick {
    //     double coordinate;
    //     std::string name;
    // };
    //
    // std::vector<tick> ticks;
    //
    // =================================
};

struct axes2d { axis x, y; };

template <>
struct std::hash<axis> {
    size_t operator()(const axis &ax) {
        size_t hash = 0;

        hash ^= std::hash<std::optional<std::string>>{}(ax.label);
        hash ^= ax.mode;
        hash ^= std::hash<std::optional<range>>{}(ax.span);

        return hash;
    }
};




inline void indented_printf(FILE *stream, int indent_level, const char *message, ...) {
    const char *INDENT_LEVEL = "    ";

    va_list args;
    va_start(args, message);

    // Yeah, I enjoy printf more than I do iostreams... So what?
    // I will reconsider when they actually make iostreams good:
    for (int i = 0; i < indent_level; ++ i)
        fprintf(stream, "%s", INDENT_LEVEL);

    vfprintf(stream, message, args);
    va_end(args);
}


struct plotting_plane;

template <>
struct std::hash<plotting_plane> {
    size_t operator()(const plotting_plane &plane);
};

struct plotting_plane {
    plotting_plane() = default;

    plotting_plane(std::string plot_name): name(std::move(plot_name)) {}

    plotting_plane(const plot &single_plot) {
        plots.emplace_back(single_plot);
        single_plot.is_shown = true;
    }

    std::vector<plot> plots;
    axes2d axes;

    std::optional<std::string> name;

    // Height can be customized via this variable:
    const char *height = "5cm";

    // Width will be decided depending on page's width:
    void draw(FILE *stream, const char *width = "\\textwidth") const;

    void generate_image(std::string output_file_name = "img.png",
        const char* width = "17cm", const char *border = "2cm", int dpi = 512);

    ~plotting_plane() {
        if (should_draw_on_destruction) {
            draw(stdout);

            // TODO: this is for caching and image generation:

            // std::filesystem::create_directory("imgs");
            // std::filesystem::create_directory("imgs/plots");

            // size_t hash = std::hash<plotting_plane>{}(*this);
            // std::stringstream ss; ss << std::hex << hash;

            // std::string file_name = "imgs/plots/" + ss.str() + ".pdf";
            // if (!std::filesystem::exists(file_name))
            //     generate_image(file_name);

            // printf("%s", file_name.c_str());
        }

        // Make sure plot's destructors aren't gonna draw them:
        for (auto &plot: plots)
            plot.is_shown = true;
    }

private:
    mutable bool should_draw_on_destruction = true;
};

inline size_t std::hash<plotting_plane>::operator()(const plotting_plane &plane) {
    size_t hash = 0;

    std::hash<axis> axis_hasher;
    hash ^= axis_hasher(plane.axes.x) ^ axis_hasher(plane.axes.y);

    for (const auto &figure: plane.plots)
        hash ^= std::hash<plot>{}(figure);

    hash ^= std::hash<std::string>{}(plane.height);
    return hash;
}



inline void plotting_plane::draw(FILE *stream, const char *width) const {
    // If draw is called explicitly, automatic drawing is going to be disabled:
    should_draw_on_destruction = false;

    indented_printf(stream, 0, "\\begin{tikzpicture}[trim axis left]\n");
    indented_printf(stream, 1, "\\begin{axis}[height=%s, width=%s,", height, width);

    bool add_legend = false;
    for (const auto &plot : plots) // TODO: extract, find out if there going to be a legend
        if (plot.name) {
            add_legend = true;
            break;
        }

    if (name) {
        fprintf(stream, " title={%s},", name->c_str());

        // Legend is generated above as legend is and it takes up precious space,
        // we need make room for it if we have title:
        if (add_legend)
            fprintf(stream, " title style={above=3ex},");
    }


    if (axes.x.span) {
        fprintf(stream, " xmin=%lf,", axes.x.span->start);
        fprintf(stream, " xmax=%lf,", axes.x.span->end);
    }

    if (axes.x.label)
        fprintf(stream, " xlabel={%s},", axes.x.label->c_str());

    if (axes.x.mode == LOG)
        fprintf(stream, " xmode=log,");


    if (axes.y.span) {
        fprintf(stream, " ymin = %lf,", axes.y.span->start);
        fprintf(stream, " ymax = %lf,", axes.y.span->end);
    }

    if (axes.y.label)
        fprintf(stream, " ylabel={%s},", axes.y.label->c_str());

    if (axes.y.mode == LOG)
        fprintf(stream, " ymode=log,");

    if (add_legend) {
        fprintf(stream, " legend style={at={(0.5,1)},anchor={south}},");
        fprintf(stream, " legend columns=-1,");
        fprintf(stream, " legend style={fill=none, draw=none},");
        fprintf(stream, " legend style={/tikz/every even column/.append style={column sep=1em}}, ");
    }

    fprintf(stream, " scale only axis, cycle list name=exotic]\n");


    // Probably extract this to a method inside plot class:
    for (const auto &plot : plots) {
        indented_printf(stream, 2, "\\addplot+[");
        if (plot.mark_size == 0)
            fprintf(stream, "mark=none");
        else {
            fprintf(stream, "mark size=%lf", plot.mark_size);
            fprintf(stream, ", mark=*");
        }

        if (plot.color)
            fprintf(stream, ", color={%s}", plot.color->c_str());

        if (plot.interpolate == SMOOTH)
            fprintf(stream, ", smooth");
        else if (plot.interpolate == NONE)
            fprintf(stream, ", only marks");

        fprintf(stream, "] coordinates {\n");

        for (const vec2 &point: plot.points)
            indented_printf(stream, 3, "(%.17g, %.17g)\n", point.x, point.y);

        indented_printf(stream, 2, "};\n");

        if (add_legend)
            indented_printf(stream, 2, "\\addlegendentry{%s};\n",
                            plot.name? plot.name->c_str() : "" /* empty legend allows to skip entry */);
    }

    indented_printf(stream, 1, "\\end{axis}\n", height);
    indented_printf(stream, 0, "\\end{tikzpicture}\n");
}






inline plotting_plane operator+(const plot &lhs, const plot &rhs) {
    lhs.is_shown = true; // TODO: probably make this system prettier!
    rhs.is_shown = true;

    plotting_plane plane {};
    plane.plots.emplace_back(lhs);
    plane.plots.emplace_back(rhs);


    return plane;
}

inline plotting_plane& operator+(plotting_plane &plane, const plot &rhs) {
    rhs.is_shown = true;

    plane.plots.emplace_back(rhs);
    return plane;
}

// TODO: copy-pasta :(, but no idead how to support older cpp without
//       sfinae hell, so let's cope with it:
inline plotting_plane&& operator+(plotting_plane &&plane, const plot &rhs) {
    rhs.is_shown = true;

    plane.plots.emplace_back(rhs);
    return std::move(plane);
}


// And... it should work both ways...

inline plotting_plane &&operator+(const plot& rhs, plotting_plane &&plane) {
    // TODO: It will mess up the order, which is... not intuitive.
    //       But the other way it's slow. Should I do anything about it?
    return std::move(plane) + rhs;
}

inline plotting_plane &operator+(const plot &rhs, plotting_plane &plane) {
    return plane + rhs;
}


// Lazy, I know, I know. TODO: Make this right.
inline plot::~plot() {
    if (!is_shown) {
        is_shown = true; // Prevent double destruction!
        plotting_plane{} + *this;
    }
}

// For converting data generated by orgmode babel responsible for running C++:

template <typename data_type>
std::vector<vec2> convert_data(data_type &&data, int col_x = 0, int col_y = 1) {
    std::vector<vec2> points;
    for (auto row: std::forward<decltype(data)>(data)) {
        points.push_back({ row[col_x], row[col_y] });
    }

    return points;
}

template <typename data_type>
plot points_table(data_type &&data, int col_x = 0, int col_y = 1) {
    return plot { .points = convert_data(std::forward<data_type>(data), col_x, col_y) };
}


// Generating PGFPlots code, from now on it's not pretty... I warned you!











// TODO: ideally, process handling should be separated from here, but for now
//       I'm making this libarary header-only, so let's cope:

#include <unistd.h>
#include <sys/wait.h>
#include <string.h>


template <typename... arg_types>
void execute(const char *program, const char* working_directory, arg_types ...args) {
    pid_t pid = fork();

    if (pid == -1) {
        perror("fork");

        // TODO: probably should do more for error reporting
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        chdir(working_directory);

        // Suppress any kind of output to stdout:
        freopen("/dev/null", "w", stdout);
        execlp(program, program, args..., NULL);
    }

    int status;
    waitpid(pid, &status, 0);
}


// Launching latex to render desired image:

static bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

inline void plotting_plane::generate_image(std::string output_file_name,
const char* width, const char *border, int dpi) {
    std::filesystem::path path = output_file_name;
    if (!path.parent_path().empty() && !std::filesystem::exists(path.parent_path()))
        execute("mkdir", ".", path.parent_path().c_str());

    char temp_directory_name[] = "/tmp/cppgfplots-plot-XXXXXX";
    mkdtemp(temp_directory_name);

    std::string image_tex = temp_directory_name;
    image_tex += "/standalone-image.tex";

    FILE *image_tex_stream = fopen(image_tex.c_str(), "w");
    fprintf(image_tex_stream,
            "\\documentclass[tikz, border={2cm, .5cm, 1cm, .5cm}]{standalone}" "\n"
            "\\usepackage{pgfplots}"                                           "\n"
                                                                               "\n"
            "\\usepackage[utf8]{inputenc}"                                     "\n"
            "\\usepackage[T2A]{fontenc}"                                       "\n"
                                                                               "\n"
            "\\usepackage{amsmath}"                                            "\n"
            "\\usepackage{amssymb}"                                            "\n"
                                                                               "\n"
            "\\pgfplotsset{compat=1.18}"                                       "\n"
                                                                               "\n"
            "\\begin{document}"                                                "\n"
    );

    this->draw(image_tex_stream, width);

    fprintf(image_tex_stream,
            "\\end{document}"                                                  "\n"
    );

    fclose(image_tex_stream), image_tex_stream = NULL;

    execute("pdflatex", temp_directory_name, "-interaction=nonstopmode", image_tex.c_str());

    std::string image_pdf = temp_directory_name;
    image_pdf += "/standalone-image.pdf";

    if (ends_with(output_file_name, ".pdf"))
        execute("mv", ".", image_pdf.c_str(), output_file_name.c_str());
    else if (ends_with(output_file_name, ".svg"))
        execute("pdf2svg", ".", image_pdf.c_str(), output_file_name.c_str());
    else {
        execute("convert", ".", "-density",
                std::to_string(dpi).c_str(), image_pdf.c_str(), output_file_name.c_str());
    }
};


#define function(f, ...) plot::analytical([&](double x) { return f; }, __VA_ARGS__)
#define points(...) plot { .points = { __VA_ARGS__ } }



// ------------------------------- MATRIX -------------------------------

template <typename underlying_type, typename index_type = size_t>
class matrix {
public:
    matrix(int num_cols): num_cols_(num_cols), num_rows_(0), data_{} {}

    class row_proxy {
    public:
      decltype(auto) operator[](index_type col_index) {
          return matrix_.get_element(row_index_, col_index);
      }

    private:
        matrix<underlying_type> &matrix_;
        int row_index_;

        row_proxy(matrix<underlying_type, index_type> &matrix, int row_index):
            matrix_(matrix), row_index_(row_index) {}

        friend class matrix;
    };

    row_proxy operator[](int row_index) { return { *this, row_index }; }

    underlying_type& get_element(index_type row_index, index_type col_index) {
        return data_[num_cols_ * row_index + col_index];
    }

    void add_row() {
        ++ num_rows_;
        data_.resize(num_cols_ * num_rows_);
    }

    index_type cols() { return num_cols_; }
    index_type rows() { return num_rows_; }

private:
    index_type num_cols_, num_rows_;
    std::vector<underlying_type> data_;

    friend class table;
};


class table: private matrix<double> {
public:
    using matrix::operator[];
    using matrix::cols;
    using matrix::rows;
    using matrix::add_row;


    static table load_table(std::string filename) {
        std::ifstream ifs(filename);

        std::vector<std::string> column_names;

        { // Read columns from the table:

            std::string line; std::getline(ifs, line);
            std::istringstream ss(line);

            std::string word;
            while (ss >> word)
                column_names.push_back(word);
        }

        table new_table(column_names.size());
        new_table.column_names_ = column_names;

        // TODO: do not access private data

        double number = 0.0;
        while (ifs >> number)
            new_table.data_.push_back(number);

        new_table.num_rows_ = new_table.data_.size() / column_names.size();
        return new_table;
    }

    void print() {
        for (const auto &name : column_names_)
            std::cout << name << "\t";

        std::cout << "\n";

        for (int i = 0; i < rows(); ++ i) {
            for (int j = 0; j < cols(); ++ j)
                std::cout << (*this)[i][j] << "\t";

            std::cout << "\n";
        }
    }

    std::vector<vec2> to_points(int col_x = 0, int col_y = 1) {
        std::vector<vec2> points;

        for (int i = 0; i < num_rows_; ++ i)
            points.push_back({ (*this)[i][col_x], (*this)[i][col_y] });

        return points;
    }

private:
    table(int num_cols): matrix(num_cols) {}

    std::vector<std::string> column_names_;
};


template <typename data_point_type>
double average(std::vector<data_point_type> const& v, double vec2::*element){
    if(v.empty()){
        return 0;
    }

    double count = static_cast<double>(v.size());

    double sum = 0;
    for (int i = 0; i < v.size(); ++i)
        sum += v[i].*element;

    return sum / count;
}

template <typename data_point_type>
double average(std::vector<data_point_type> const& v){
    if(v.empty()){
        return 0;
    }

    double count = static_cast<double>(v.size());

    double sum = 0;
    for (int i = 0; i < v.size(); ++i)
        sum += v[i];

    return sum / count;
}

inline double square(double value) { return value * value; }

struct mnk_results {
    double x_avg;
    double sx;

    double y_avg;
    double sy;

    double rxy;

    double a;
    double da;

    double b;
    double db;
};

inline mnk_results mnk(const std::vector<vec2> &points) {
    std::vector<double> x_squared;
    std::vector<double> y_squared;
    std::vector<double> xy;

    for (const auto &point : points) {
        x_squared.push_back(point.x * point.x);
        y_squared.push_back(point.y * point.y);

        xy.push_back(point.x * point.y);
    }

    double sx = average(x_squared) - square(average(points, &vec2::x));
    double sy = average(y_squared) - square(average(points, &vec2::y));

    double rxy = average(xy) - (average(points, &vec2::x) * average(points, &vec2::y));

    double a = rxy / sx;
    double da = sqrt(1. / (points.size() - 2) * (sy / sx - square(a)));

    double b = average(points, &vec2::y) - a * average(points, &vec2::x);
    double db = da * sqrt(sx + square(average(points, &vec2::x)));

    return {
        average(points, &vec2::x), sx,
        average(points, &vec2::y), sy,
        rxy, a, da, b, db
    };
}


inline void spectre(table &tbl) {
    auto plot_cols = [](plotting_plane &plane, table &tbl, int x, int y, const char *color) {
        auto points = tbl.to_points(x, y);
        std::vector<vec2> reduced_points;

        for (int i = 0; i < points.size(); i += 15)
            reduced_points.push_back(points[i]);

        plot plt { reduced_points };

        plt.mark_size = 0;
        plt.color = color;

        plane + plt;
    };

    plotting_plane plane;
    plot_cols(plane, tbl, 0, 1, "red");
    plot_cols(plane, tbl, 0, 2, "green");
    plot_cols(plane, tbl, 0, 3, "teal");
    plot_cols(plane, tbl, 0, 4, "blue");

    plane.axes.x.label = "Длина волны, \\AA";
}
