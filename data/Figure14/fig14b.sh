cd ../../build/Release

chmod +x gridgen

./gridgen ../../data/Figure14/grid_1.json ../../data/Figure14/figure14.json -t 0.001 -o "CSG" --tree ../../data/Figure14/figure14_tree.json
