//
//  io.cpp
//  adaptive_mesh_refinement
//
//  Created by Yiwen Ju on 8/1/24.
//

#include <mtet/io.h>

#include <AdaptiveGrid/io.h>

using namespace AdaptiveGrid;

namespace {

///
/// Save tet grid to JSON file.
///
/// @param[in] filename  The name of the output file.
/// @param[in] mesh      The tet grid.
///
/// @note Since JSON files is in ASCII, it is not recommended for large meshes.
///
void save_tet_mesh_json(const std::string& filename, const mtet::MTetMesh& mesh)
{
    using namespace mtet;
    std::vector<std::array<double, 3>> vertices((int)mesh.get_num_vertices());
    std::vector<std::array<size_t, 4>> tets((int)mesh.get_num_tets());
    ankerl::unordered_dense::map<uint64_t, size_t> vertex_tag_map;
    vertex_tag_map.reserve(mesh.get_num_vertices());
    int counter = 0;
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data) {
        // TODO: The +1 and later -1 is redundant...
        size_t vertex_tag = vertex_tag_map.size() + 1;
        vertex_tag_map[value_of(vid)] = vertex_tag;
        vertices[counter] = {data[0], data[1], data[2]};
        counter++;
    });
    counter = 0;
    mesh.seq_foreach_tet([&](TetId, std::span<const VertexId, 4> data) {
        tets[counter] = {
            vertex_tag_map[value_of(data[0])] - 1,
            vertex_tag_map[value_of(data[1])] - 1,
            vertex_tag_map[value_of(data[2])] - 1,
            vertex_tag_map[value_of(data[3])] - 1};
        counter++;
    });
    if (std::filesystem::exists(filename.c_str())) {
        std::filesystem::remove(filename.c_str());
    }
    using json = nlohmann::json;
    std::ofstream fout(filename.c_str(), std::ios::app);
    json jOut;
    jOut.push_back(json(vertices));
    jOut.push_back(json(tets));
    fout << jOut.dump(4, ' ', true, json::error_handler_t::replace) << std::endl;
    fout.close();
}

} // namespace

mtet::MTetMesh AdaptiveGrid::load_tet_mesh(const std::string& filename)
{
    using json = nlohmann::json;
    std::ifstream fin(filename.c_str());
    if (!fin) {
        throw std::runtime_error("tet mesh file not exist!");
    }
    json data;
    fin >> data;
    fin.close();
    // if the tet grid is specified by resolution and bounding box
    if (data.contains("resolution")) {
        size_t num_resolution = data["resolution"].size();
        assert(num_resolution <= 3 && num_resolution > 0);
        size_t res = data["resolution"][0].get<size_t>();
        std::array<size_t, 3> resolution = {res, res, res};
        for (size_t i = 0; i < num_resolution; i++) {
            resolution[i] = data["resolution"][i].get<size_t>();
        }
        assert(data.contains("bbox_min"));
        assert(data["bbox_min"].size() == 3);
        std::array<double, 3> bbox_min{0, 0, 0};
        for (size_t i = 0; i < 3; i++) {
            bbox_min[i] = data["bbox_min"][i].get<double>();
        }
        assert(data.contains("bbox_max"));
        assert(data["bbox_max"].size() == 3);
        std::array<double, 3> bbox_max{1, 1, 1};
        for (size_t i = 0; i < 3; i++) {
            bbox_max[i] = data["bbox_max"][i].get<double>();
        }
        AdaptiveGrid::GridStyle style = AdaptiveGrid::TET6;
        if (data.contains("style")) {
            auto style_str = data["style"].get<std::string>();
            if (style_str == "TET5") {
                style = AdaptiveGrid::TET5;
            } else if (style_str == "TET6") {
                style = AdaptiveGrid::TET6;
            } else {
                throw std::runtime_error("unknown grid style!");
            }
        }
        return AdaptiveGrid::generate_tet_grid(resolution, bbox_min, bbox_max, style);
    }
    // vertices
    std::vector<std::array<double, 3>> pts;
    pts.resize(data[0].size());
    for (size_t j = 0; j < pts.size(); j++) {
        for (size_t k = 0; k < 3; k++) {
            pts[j][k] = data[0][j][k].get<double>();
        }
    }
    // tets
    std::vector<std::array<size_t, 4>> tets;
    tets.resize(data[1].size());
    for (size_t j = 0; j < tets.size(); j++) {
        for (size_t k = 0; k < 4; k++) {
            tets[j][k] = data[1][j][k].get<size_t>();
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

void AdaptiveGrid::save_tet_mesh(const std::string& filename, const mtet::MTetMesh& mesh)
{
    const std::string ext = filename.substr(filename.find_last_of("."));
    if (ext == ".json") {
        save_tet_mesh_json(filename, mesh);
    } else if (ext == ".msh") {
        mtet::save_mesh(filename, mesh);
    } else {
        throw std::runtime_error("unknown file format!");
    }
}
