//
// Created by Yiwen Ju on 8/4/24.
//
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include "csg.h"
#include "grid_refine.h"
#include "refine_crit.h"
#include "tet_quality.h"
#include "timer.h"

#include <catch2/catch.hpp>

/// tests:
/// examples with known structure
/// IA example: 18 sphers
/// CSG example: 20 tori (10 tori - 10 tori)
/// MI examples: 18 spheres


double alpha = std::numeric_limits<double>::infinity();
int max_elements = std::numeric_limits<int>::max();
double smallest_edge_length = 0;
bool curve_network = false;

TEST_CASE("grid generation of implicit arrangement on known examples", "[IA][examples]")
{
    std::string function_file;
    double threshold;
    mtet::MTetMesh grid;
    llvm_vecsmall::SmallVector<csg_unit, 20> csg_tree = {};

    SECTION("18 spheres")
    {
        // parse configurations
        threshold = 0.0001;
        grid = mtet::load_mesh(std::string(TEST_FILE) + "/grid/cube6.msh");
        std::string function_file = std::string(TEST_FILE) + "/function_examples/18-sphere.json";
        AdaptiveGrid::tet_metric metric_list;
        std::array<double, timer_amount> profileTimer = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<std::unique_ptr<ImplicitFunction<double>>> functions;
        load_functions(function_file, functions);
        const size_t funcNum = functions.size();
        auto implicit_func = [&](std::span<const AdaptiveGrid::Scalar, 3> data, size_t funcNum) {
            llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20> vertex_eval(funcNum);
            for (size_t funcIter = 0; funcIter < funcNum; funcIter++) {
                auto& func = functions[funcIter];
                Eigen::Vector4d eval;
                eval[0] =
                    func->evaluate_gradient(data[0], data[1], data[2], eval[1], eval[2], eval[3]);
                vertex_eval[funcIter] = eval;
            }
            return vertex_eval;
        };
        auto csg_func = [&](llvm_vecsmall::SmallVector<std::array<double, 2>, 20> funcInt) {
            std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>> null_csg = {
                {},
                {}};
            return null_csg;
        };

        // start testing
        bool success = gridRefine(
            AdaptiveGrid::IA,
            curve_network,
            threshold,
            alpha,
            max_elements,
            funcNum,
            implicit_func,
            csg_func,
            grid,
            metric_list,
            profileTimer);
        REQUIRE(success);

        // check
        REQUIRE(metric_list.total_tet == 1510932);
        REQUIRE(metric_list.active_tet == 888666);
        REQUIRE(metric_list.two_func_check == 93455);
        REQUIRE(metric_list.three_func_check == 1576);
    }
}

TEST_CASE("grid generation of CSG on known examples", "[CSG][examples]")
{
    std::string function_file;
    double threshold;
    llvm_vecsmall::SmallVector<csg_unit, 20> csg_tree = {};
    mtet::MTetMesh grid;

    SECTION("20 tori")
    {
        // parse configurations
        threshold = 0.03;
        grid = AdaptiveGrid::load_tet_mesh(std::string(TEST_FILE) + "/Figure21/grid_1.json");
        mtet::save_mesh("init.msh", grid);
        grid = mtet::load_mesh("init.msh");
        std::string function_file = std::string(TEST_FILE) + "/Figure21/csg_examples_3.json";
        std::string csg_file = std::string(TEST_FILE) + "/Figure21/csg_examples_3_tree.json";
        load_csgTree(csg_file, csg_tree);
        AdaptiveGrid::tet_metric metric_list;
        std::array<double, timer_amount> profileTimer = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<std::unique_ptr<ImplicitFunction<double>>> functions;
        load_functions(function_file, functions);
        const size_t funcNum = functions.size();

        auto implicit_func = [&](std::span<const AdaptiveGrid::Scalar, 3> data, size_t funcNum) {
            llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20> vertex_eval(funcNum);
            for (size_t funcIter = 0; funcIter < funcNum; funcIter++) {
                auto& func = functions[funcIter];
                Eigen::Vector4d eval;
                eval[0] =
                    func->evaluate_gradient(data[0], data[1], data[2], eval[1], eval[2], eval[3]);
                vertex_eval[funcIter] = eval;
            }
            return vertex_eval;
        };
        auto csg_func = [&](llvm_vecsmall::SmallVector<std::array<double, 2>, 20> funcInt) {
            if (csg_file == "") {
                throw std::runtime_error("ERROR: no csg file provided");
                std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>> null_csg = {
                    {},
                    {}};
                return null_csg;
            } else {
                return iterTree(csg_tree, 1, funcInt);
            }
        };
        // start testing
        bool success = gridRefine(
            AdaptiveGrid::CSG,
            curve_network,
            threshold,
            alpha,
            max_elements,
            funcNum,
            implicit_func,
            csg_func,
            grid,
            metric_list,
            profileTimer);
        REQUIRE(success);

        // check
        REQUIRE(metric_list.total_tet == 96174);
        REQUIRE(metric_list.active_tet == 47485);
        REQUIRE(metric_list.two_func_check == 58897);
        REQUIRE(metric_list.three_func_check == 9836);
    }
}

TEST_CASE("grid generation of material interface on known examples", "[MI][examples]")
{
    std::string function_file;
    double threshold;
    mtet::MTetMesh grid;
    llvm_vecsmall::SmallVector<csg_unit, 20> csg_tree = {};
    auto csg_func = [&](llvm_vecsmall::SmallVector<std::array<double, 2>, 20> funcInt) {
        std::pair<std::array<double, 2>, llvm_vecsmall::SmallVector<int, 20>> null_csg = {{}, {}};
        return null_csg;
    };

    SECTION("18 spheres")
    {
        // parse configurations
        threshold = 0.0001;
        grid = mtet::load_mesh(std::string(TEST_FILE) + "/grid/cube6.msh");
        std::string function_file = std::string(TEST_FILE) + "/function_examples/18-sphere.json";
        AdaptiveGrid::tet_metric metric_list;
        std::array<double, timer_amount> profileTimer = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<std::unique_ptr<ImplicitFunction<double>>> functions;
        load_functions(function_file, functions);
        const size_t funcNum = functions.size();
        auto implicit_func = [&](std::span<const AdaptiveGrid::Scalar, 3> data, size_t funcNum) {
            llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20> vertex_eval(funcNum);
            for (size_t funcIter = 0; funcIter < funcNum; funcIter++) {
                auto& func = functions[funcIter];
                Eigen::Vector4d eval;
                eval[0] =
                    func->evaluate_gradient(data[0], data[1], data[2], eval[1], eval[2], eval[3]);
                vertex_eval[funcIter] = eval;
            }
            return vertex_eval;
        };
        // start testing
        bool success = gridRefine(
            AdaptiveGrid::MI,
            curve_network,
            threshold,
            alpha,
            max_elements,
            funcNum,
            implicit_func,
            csg_func,
            grid,
            metric_list,
            profileTimer);
        REQUIRE(success);

        // check
        REQUIRE(metric_list.total_tet == 905468);
        REQUIRE(metric_list.active_tet == 405118);
        REQUIRE(metric_list.two_func_check == 11227);
        REQUIRE(metric_list.three_func_check == 156);
    }
}

