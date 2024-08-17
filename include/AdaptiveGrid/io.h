//
//  io.h
//  adaptive_mesh_refinement
//
//  Created by Yiwen Ju on 8/1/24.
//

#pragma once

#include <AdaptiveGrid/grid.h>

#include <ankerl/unordered_dense.h>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include "adaptive_grid_gen.h"
#include "timer.h"

namespace AdaptiveGrid {

///
/// Load tet grid from JSON file.
///
/// An example JSON file is as follows:
/// ```
///   {
///       "resolution": [10, 10, 10],
///       "bbox_min": [-1, -1, -1],
///       "bbox_max": [1, 1, 1]
///   }
/// ```
///
/// @param[in] filename  The name of the input file.
/// @return  The tet grid.
///
mtet::MTetMesh load_tet_mesh(const std::string& filename);

///
/// Save tet grid to file. Both .json and .msh formats are supported.
///
/// @param[in] filename  The name of the output file.
/// @param[in] mesh      The tet grid.
///
void save_tet_mesh(const std::string& filename, const mtet::MTetMesh& mesh);

} // namespace AdaptiveGrid
