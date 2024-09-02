#include <CLI/CLI.hpp>
#include <limits>
#include <span>

#include <AdaptiveGrid/grid.h>
#include <AdaptiveGrid/io.h>

#include "csg.h"
#include "grid_refine.h"
#include "timer.h"


bool save_timings(
    const std::string& filename,
    const std::array<std::string, timer_amount>& time_label,
    const std::array<double, timer_amount>& timings)
{
    using json = nlohmann::json;
    std::ofstream fout(filename.c_str(), std::ios::app);
    // fout.open(filename.c_str(),std::ios::app);
    json jOut;
    for (size_t i = 0; i < timings.size(); ++i) {
        jOut[time_label[i]] = timings[i];
    }
    //
    fout << jOut << std::endl;
    fout.close();
    return true;
}


bool save_function_json(
    const std::string& filename,
    const mtet::MTetMesh mesh,
    ankerl::unordered_dense::map<uint64_t, llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>>
        vertex_func_grad_map,
    const size_t funcNum)
{
    using namespace mtet;
    std::vector<std::vector<double>> values(funcNum);
    for (size_t funcIter = 0; funcIter < funcNum; funcIter++) {
        values[funcIter].reserve(((int)mesh.get_num_vertices()));
    }
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const AdaptiveGrid::Scalar, 3> data) {
        llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20> func_gradList(funcNum);
        func_gradList = vertex_func_grad_map[value_of(vid)];
        for (size_t funcIter = 0; funcIter < funcNum; funcIter++) {
            values[funcIter].push_back(func_gradList[funcIter][0]);
        }
    });
    if (std::filesystem::exists(filename.c_str())) {
        std::filesystem::remove(filename.c_str());
    }
    using json = nlohmann::json;
    std::ofstream fout(filename.c_str(), std::ios::app);
    json jOut;
    for (size_t funcIter = 0; funcIter < funcNum; funcIter++) {
        json jFunc;
        jFunc["type"] = "customized";
        jFunc["value"] = values[funcIter];
        jOut.push_back(jFunc);
    }
    fout << jOut.dump(4, ' ', true, json::error_handler_t::replace) << std::endl;
    fout.close();
    return true;
}


bool save_metrics(
    const std::string& filename,
    const std::array<std::string, 6>& tet_metric_labels,
    const AdaptiveGrid::tet_metric metric_list)
{
    using json = nlohmann::json;
    std::ofstream fout(filename.c_str(), std::ios::app);
    json jOut;
    jOut[tet_metric_labels[0]] = metric_list.total_tet;
    jOut[tet_metric_labels[1]] = metric_list.active_tet;
    jOut[tet_metric_labels[2]] = metric_list.min_radius_ratio;
    jOut[tet_metric_labels[3]] = metric_list.active_radius_ratio;
    jOut[tet_metric_labels[4]] = metric_list.two_func_check;
    jOut[tet_metric_labels[5]] = metric_list.three_func_check;
    fout << jOut << std::endl;
    fout.close();
    return true;
}


int main(int argc, const char* argv[])
{
    struct
    {
        std::string grid_file;
        std::string function_file;
        double threshold;
        double alpha = std::numeric_limits<double>::infinity();
        int max_elements = std::numeric_limits<int>::max();
        double smallest_edge_length = 0;
        std::string method = "IA";
        std::string csg_file;
        bool bfs = false;
        bool dfs = false;
        bool curve_network = false;
        bool discretize_later = false;
    } args;
    CLI::App app{"Longest Edge Bisection Refinement"};
    app.add_option("grid", args.grid_file, "Initial grid file")->required();
    app.add_option("function", args.function_file, "Implicit function file")->required();
    app.add_option("-t,--threshold", args.threshold, "Threshold value");
    app.add_option("-a,--alpha", args.alpha, "Alpha value");
    app.add_option("-o,--option", args.method, "Options of implicit manifold");
    app.add_option("--tree", args.csg_file, "CSG Tree file");
    app.add_option("-m,--max-elements", args.max_elements, "Maximum number of elements");
    app.add_option("-s,--shortest-edge", args.smallest_edge_length, "Shortest edge length");
    app.add_option(
        "-d,--discretize",
        args.discretize_later,
        "Save the grid file and function values for discretizing them later");
    app.add_option("-c, --curve_network", args.curve_network, "Generate Curve Network only");
    CLI11_PARSE(app, argc, argv);
    // Read initial grid
    mtet::MTetMesh grid;
    if (args.grid_file.find(".json") != std::string::npos) {
        grid = AdaptiveGrid::load_tet_mesh(args.grid_file);
    } else {
        grid = mtet::load_mesh(args.grid_file);
    }

    std::string function_file = args.function_file;
    double threshold = args.threshold;
    AdaptiveGrid::Mode mode;
    llvm_vecsmall::SmallVector<csg_unit, 20> csg_tree = {};

    if (args.method == "IA") {
        mode = AdaptiveGrid::IA;
    }
    if (args.method == "CSG") {
        mode = AdaptiveGrid::CSG;
        load_csgTree(args.csg_file, csg_tree);
    }
    if (args.method == "MI") {
        mode = AdaptiveGrid::MI;
    }

    /// Read implicit function
    std::vector<std::unique_ptr<ImplicitFunction<double>>> functions;
    load_functions(function_file, functions);
    const size_t funcNum = functions.size();

    ///
    /// the lambda function for function evaluations
    ///  @param[in] data            The 3D coordinate
    ///  @param[in] funcNum         The number of functions
    ///
    ///  @return        A vector of `Eigen::RowVector4d`.The vector size is the function number. Each eigen vector represents the value at 0th index and gradients at {1, 2, 3} index.
    auto implicit_func = [&](std::span<const AdaptiveGrid::Scalar, 3> data, size_t funcNum) {
        llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20> vertex_eval(funcNum);
        for (size_t funcIter = 0; funcIter < funcNum; funcIter++) {
            auto& func = functions[funcIter];
            Eigen::Vector4d eval;
            eval[0] = func->evaluate_gradient(data[0], data[1], data[2], eval[1], eval[2], eval[3]);
            vertex_eval[funcIter] = eval;
        }
        return vertex_eval;
    };

    ///
    /// the lambda function for csg tree iteration/evaluation.
    /// @param[in] funcInt          Given an input of value range std::array<double, 2> for an arbitrary number of functions
    /// @return   A value range of this CSG operation in a form of `std::array<double, 2>` and a list of active function in a form of    `llvm_vecsmall::SmallVector<int, 20>>`
    ///
    auto csg_func = [&](llvm_vecsmall::SmallVector<std::array<double, 2>, 20> funcInt) {
        if (args.csg_file == "") {
            throw std::runtime_error("ERROR: no csg file provided");
            std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>> null_csg = {
                {},
                {}};
            return null_csg;
        } else {
            return iterTree(csg_tree, 1, funcInt);
        }
    };

    // perform main grid refinement algorithm:
    AdaptiveGrid::tet_metric metric_list;
    // an array of 10 timings: {total time getting the multiple indices, total time,time spent on
    // single function, time spent on double functions, time spent on triple functions time spent on
    // double functions' zero crossing test, time spent on three functions' zero crossing test,
    // total subdivision time, total evaluation time,total splitting time}
    std::array<double, timer_amount> profileTimer = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (!gridRefine(
            mode,
            args.curve_network,
            args.threshold,
            args.alpha,
            args.max_elements,
            funcNum,
            implicit_func,
            csg_func,
            grid,
            metric_list,
            profileTimer)) {
        throw std::runtime_error("ERROR: unsuccessful grid refinement");
    }
    // save timing records
    save_timings("timings.json", time_label, profileTimer);
    // profiled time(see details in time.h) and profiled number of calls to zero
    for (int i = 0; i < profileTimer.size(); i++) {
        timeProfileName time_type = static_cast<timeProfileName>(i);
        std::cout << time_label[i] << ": " << profileTimer[i] << std::endl;
    }
    // save tet metrics
    save_metrics("stats.json", tet_metric_labels, metric_list);


    if (args.discretize_later) {
        /// save the grid output for discretization tool
        AdaptiveGrid::save_tet_mesh("grid.json", grid);
        /// save the grid output for isosurfacing tool
        save_function_json("function_value.json", grid, metric_list.vertex_func_grad_map, funcNum);
        /// write grid and active tets
        mtet::save_mesh("tet_grid.msh", grid);
        mtet::save_mesh("active_tets.msh", grid, std::span<mtet::TetId>(metric_list.activeTetId));
    }
    return 0;
}
