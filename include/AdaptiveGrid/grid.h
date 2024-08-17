#pragma once

#include <mtet/mtet.h>

#include <array>

namespace AdaptiveGrid {

enum GridStyle {
    TET5, // Split each hexahedron into 5 tetrahedrons
    TET6 // Split each hexahedron into 6 tetrahedrons
};

///
/// Generate tet grid.
///
/// This function first creates a regular grid of hexahedrons, then splits each
/// hexahedron into tetrahedrons according to the given style.
///
/// @param resolution Resolution of the grid in each dimension
/// @param bbox_min   Minimum corner of the bounding box
/// @param bbox_max   Maximum corner of the bounding box
/// @param style      How to split each hexahedron
///
/// @return Tet grid mesh
///
mtet::MTetMesh generate_tet_grid(
    const std::array<size_t, 3>& resolution,
    const std::array<double, 3>& bbox_min,
    const std::array<double, 3>& bbox_max,
    GridStyle style = TET5);

} // namespace AdaptiveGrid
