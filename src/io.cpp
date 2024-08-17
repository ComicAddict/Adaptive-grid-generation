//
//  init.cpp
//  adaptive_mesh_refinement
//
//  Created by Yiwen Ju on 8/1/24.
//

#include "io.h"

mtet::MTetMesh load_tet_mesh(const std::string &filename) {
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
    for (auto &v: pts) {
        vertex_ids.push_back(mesh.add_vertex(v[0], v[1], v[2]));
    }
    for (auto &t: tets) {
        mesh.add_tet(vertex_ids[t[0]], vertex_ids[t[1]], vertex_ids[t[2]], vertex_ids[t[3]]);
    }
    return mesh;
}


bool save_mesh_json(const std::string& filename,
                    const mtet::MTetMesh mesh)
{
    std::vector<std::array<double, 3>> vertices((int)mesh.get_num_vertices());
    std::vector<std::array<size_t, 4>> tets((int)mesh.get_num_tets());
    ankerl::unordered_dense::map<uint64_t, size_t> vertex_tag_map;
    vertex_tag_map.reserve(mesh.get_num_vertices());
    int counter = 0;
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data){
        size_t vertex_tag = vertex_tag_map.size() + 1;
        vertex_tag_map[value_of(vid)] = vertex_tag;
        vertices[counter] = {data[0], data[1], data[2]};
        counter ++;
    });
    counter = 0;
    mesh.seq_foreach_tet([&](TetId, std::span<const VertexId, 4> data) {
        tets[counter] = {vertex_tag_map[value_of(data[0])] - 1, vertex_tag_map[value_of(data[1])] - 1, vertex_tag_map[value_of(data[2])] - 1, vertex_tag_map[value_of(data[3])] - 1};
        counter ++;
    });
    if (std::filesystem::exists(filename.c_str())){
        std::filesystem::remove(filename.c_str());
    }
    using json = nlohmann::json;
    std::ofstream fout(filename.c_str(),std::ios::app);
    json jOut;
    jOut.push_back(json(vertices));
    jOut.push_back(json(tets));
    fout << jOut.dump(4, ' ', true, json::error_handler_t::replace) << std::endl;
    fout.close();
    return true;
}

bool save_function_json(const std::string& filename,
                        const mtet::MTetMesh mesh,
                        ankerl::unordered_dense::map<uint64_t, llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20>> vertex_func_grad_map,
                        const size_t funcNum)
{
    std::vector<std::vector<double>> values(funcNum);
    for (size_t funcIter = 0; funcIter <  funcNum; funcIter++){
        values[funcIter].reserve(((int)mesh.get_num_vertices()));
    }
    mesh.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data){
        llvm_vecsmall::SmallVector<Eigen::RowVector4d, 20> func_gradList(funcNum);
        func_gradList = vertex_func_grad_map[value_of(vid)];
        for (size_t funcIter = 0; funcIter < funcNum; funcIter++){
            values[funcIter].push_back(func_gradList[funcIter][0]);
        }
    });
    if (std::filesystem::exists(filename.c_str())){
        std::filesystem::remove(filename.c_str());
    }
    using json = nlohmann::json;
    std::ofstream fout(filename.c_str(),std::ios::app);
    json jOut;
    for (size_t funcIter = 0; funcIter <  funcNum; funcIter++){
        json jFunc;
        jFunc["type"] = "customized";
        jFunc["value"] = values[funcIter];
        jOut.push_back(jFunc);
    }
    fout << jOut.dump(4, ' ', true, json::error_handler_t::replace) << std::endl;
    fout.close();
    return true;
}

bool save_timings(const std::string& filename,
                  const std::array<std::string, timer_amount>& time_label,
                  const std::array<double, timer_amount>& timings)
{
    using json = nlohmann::json;
    std::ofstream fout(filename.c_str(),std::ios::app);
    //fout.open(filename.c_str(),std::ios::app);
    json jOut;
    for (size_t i = 0; i < timings.size(); ++i) {
        jOut[time_label[i]] = timings[i];
    }
    //
    fout << jOut << std::endl;
    fout.close();
    return true;
}

bool save_metrics(const std::string& filename,
                  const std::array<std::string, 6>& tet_metric_labels,
                  const tet_metric metric_list)
{
    using json = nlohmann::json;
    std::ofstream fout(filename.c_str(),std::ios::app);
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
