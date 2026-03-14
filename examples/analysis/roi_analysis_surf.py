"""
ROI analysis of the electric field from a simulation using an atlas.

calculates the mean electric field in a gray matter ROI defined using an atlas

Copyright (c) 2019 SimNIBS developers. Licensed under the GPL v3.
"""

import os
import numpy as np
import simnibs

## Input ##

# Read the simulation result mapped to the gray matter surface
gm_surf = simnibs.read_msh(
    os.path.join("tdcs_simu", "subject_overlays", "ernie_TDCS_1_scalar_central.msh")
)

# Load the atlas and define the brain region of interest
atlas = simnibs.atlas2subject("m2m_ernie", "HCP_MMP1", split_labels=True)
region = "V2"
# we need to concatenate lh and rh masks. In the mesh file, the order is lh
# first, then rh
hemi = "lh"
roi_lh = atlas[hemi][region]
roi_rh = np.zeros_like(atlas["rh"][region])
roi = np.concatenate([roi_lh, roi_rh])

# plot the roi
gm_surf.add_node_field(roi, "ROI")
gm_surf.view(visible_fields="ROI").show()

# calculate the node areas, we will use those later for averaging
node_areas = gm_surf.nodes_areas()

# finally, calculate the mean of the field strength
field_name = "E_magn"
mean_magnE = np.average(gm_surf.field[field_name][roi], weights=node_areas[roi])
print(f"mean {field_name} in {hemi} {region} : {mean_magnE}")
