#include <AdaptiveGrid/grid.h>

#include <array>
#include <cassert>
#include <exception>
#include <vector>

mtet::MTetMesh AdaptiveGrid::generate_tet_grid(
    const std::array<size_t, 3>& resolution,
    const std::array<double, 3>& bbox_min,
    const std::array<double, 3>& bbox_max,
    GridStyle style)
{
    assert(resolution[0] > 0 && resolution[1] > 0 && resolution[2] > 0);
    const size_t N0 = resolution[0] + 1;
    const size_t N1 = resolution[1] + 1;
    const size_t N2 = resolution[2] + 1;
    std::vector<std::array<double, 3>> pts(N0 * N1 * N2);
    auto compute_coordinate = [&](double t, size_t i) {
        return t * (bbox_max[i] - bbox_min[i]) + bbox_min[i];
    };
    // vertices
    for (size_t i = 0; i < N0; i++) {
        double x = compute_coordinate(double(i) / double(N0 - 1), 0);
        for (size_t j = 0; j < N1; j++) {
            double y = compute_coordinate(double(j) / double(N1 - 1), 1);
            for (size_t k = 0; k < N2; k++) {
                double z = compute_coordinate(double(k) / double(N2 - 1), 2);

                size_t idx = i * N1 * N2 + j * N2 + k;
                pts[idx] = {x, y, z};
            }
        }
    }
    // tets
    std::vector<std::array<size_t, 4>> tets;
    size_t num_tet_per_cell = 0;
    if (style == TET5) {
        num_tet_per_cell = 5;
    } else if (style == TET6) {
        num_tet_per_cell = 6;
    } else {
        throw std::runtime_error("unknown grid style!");
    }
    tets.resize(resolution[0] * resolution[1] * resolution[2] * num_tet_per_cell);
    for (size_t i = 0; i < resolution[0]; i++) {
        for (size_t j = 0; j < resolution[1]; j++) {
            for (size_t k = 0; k < resolution[2]; k++) {
                size_t idx =
                    (i * resolution[1] * resolution[2] + j * resolution[2] + k) * num_tet_per_cell;
                size_t v0 = i * N1 * N2 + j * N2 + k;
                size_t v1 = (i + 1) * N1 * N2 + j * N2 + k;
                size_t v2 = (i + 1) * N1 * N2 + (j + 1) * N2 + k;
                size_t v3 = i * N1 * N2 + (j + 1) * N2 + k;
                size_t v4 = i * N1 * N2 + j * N2 + k + 1;
                size_t v5 = (i + 1) * N1 * N2 + j * N2 + k + 1;
                size_t v6 = (i + 1) * N1 * N2 + (j + 1) * N2 + k + 1;
                size_t v7 = i * N1 * N2 + (j + 1) * N2 + k + 1;
                switch (style) {
                case TET5:
                    if ((i + j + k) % 2 == 0) {
                        tets[idx] = {v4, v6, v1, v3};
                        tets[idx + 1] = {v6, v3, v4, v7};
                        tets[idx + 2] = {v1, v3, v0, v4};
                        tets[idx + 3] = {v3, v1, v2, v6};
                        tets[idx + 4] = {v4, v1, v6, v5};
                    } else {
                        tets[idx] = {v7, v0, v2, v5};
                        tets[idx + 1] = {v2, v3, v0, v7};
                        tets[idx + 2] = {v5, v7, v0, v4};
                        tets[idx + 3] = {v7, v2, v6, v5};
                        tets[idx + 4] = {v0, v1, v2, v5};
                    }
                    break;
                case TET6:
                    //{{0, 4, 6, 7},
                    // {6, 0, 5, 4},
                    // {1, 0, 5, 6},
                    // {1, 2, 0, 6},
                    // {0, 6, 2, 3},
                    // {6, 3, 0, 7}}
                    tets[idx] = {v0, v4, v6, v7};
                    tets[idx + 1] = {v6, v0, v5, v4};
                    tets[idx + 2] = {v1, v0, v5, v6};
                    tets[idx + 3] = {v1, v2, v0, v6};
                    tets[idx + 4] = {v0, v6, v2, v3};
                    tets[idx + 5] = {v6, v3, v0, v7};
                    break;
                }
            }
        }
    }
    // build mesh
    mtet::MTetMesh mesh;
    std::vector<mtet::VertexId> vertex_ids;
    vertex_ids.reserve(pts.size());
    for (auto& v : pts) {
        vertex_ids.push_back(mesh.add_vertex(v[0], v[1], v[2]));
    }
    for (auto& t : tets) {
        mesh.add_tet(vertex_ids[t[0]], vertex_ids[t[1]], vertex_ids[t[2]], vertex_ids[t[3]]);
    }
    return mesh;
}
