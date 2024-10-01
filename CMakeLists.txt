cmake_minimum_required(VERSION 3.13.4)

project(HierarchicalGeodesicModeling)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/KitwareMedical/HierarchicalGeodesicModeling")
set(EXTENSION_CATEGORY "Shape Analysis")
set(EXTENSION_CONTRIBUTORS "Ye Han, James Fishbaugh (Kitware)")
set(EXTENSION_DESCRIPTION "This 3D Slicer module is for computation and visualization/analysis of longtiudinal shape models.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/KitwareMedical/HierarchicalGeodesicModeling/refs/heads/main/HierarchicalGeodesicModeling.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/KitwareMedical/HierarchicalGeodesicModeling/refs/heads/main/HGM_Screenshot.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(HGMComputation)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
