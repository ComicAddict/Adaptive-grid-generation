DIRECTORY="../../build"
FILENAME="gridgen"

# Use the find command to search for the file
GRIDGEN=$(find "$DIRECTORY" -type f -name "$FILENAME" -print -quit)

# Check if the file was found
if [ -z "$GRIDGEN" ]; then
  echo "File not found"
else
  "$GRIDGEN" ../../data/grid/cube6.msh ../../data/Figure1/10-wikiBall.json -t 0.0005 "-o" "CSG" "--tree" ../../data/Figure1/10-wikiBall-tree.json --discretize 1
fi
