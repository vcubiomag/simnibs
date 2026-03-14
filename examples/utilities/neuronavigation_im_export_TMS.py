"""Example on how to im- or export TMS coil positions in Python
Run with:

simnibs_python neuronavigation_im_export_TMS.py

Copyright (c) 2025 SimNIBS developers. Licensed under the GPL v3.
"""

import os
import numpy as np
from simnibs import sim_struct, opt_struct, run_simnibs, localite, brainsight, ant


# =======================================
# Exporting positions of a TMS simulation
# =======================================
# Define a TMS simulation with 2 coil positions
S = sim_struct.SESSION()
S.subpath = "m2m_ernie"  # m2m-folder of the subject
S.pathfem = "tms_simu_forexport"

tmslist = S.add_tmslist()
tmslist.fnamecoil = os.path.join("legacy_and_other", "Magstim_70mm_Fig8.ccd")  

pos = tmslist.add_position()
pos.centre = "C3"
pos.pos_ydir = "CP3"
pos.distance = 4

pos2 = tmslist.add_position()
pos2.centre = "O1"
pos2.pos_ydir = "O2"
pos2.distance = 4

# Run Simulation
run_simnibs(S)
# The simulation updates the two positions to contain 4x4 position
# matrices that can be exported to use by neuronavigation systems

# Export positions
localite().write(tmslist, 'coilpositions_localite_InstrumentMarker.xml')
brainsight().write(tmslist, 'coilpositions_brainsight.txt')
ant().write(tmslist, 'coilpositions_ant.mrk', subpath=S.subpath)
# Note: See the documentation for further information about importing
# the results into the neuronavigation systems


# ================================================
# Importing positions from neuronavigation systems
# ================================================
# please run the above code to create the files

tms_list_loc = localite().read('coilpositions_localite_InstrumentMarker.xml')

tms_list_brains, tms_list_brains_samples = brainsight().read('coilpositions_brainsight.txt')
# Note: two lists are returned, one for targets and one for samples. 
# In this example, the sample list does not contain any positions and is not used further

tms_list_ant = ant().read('coilpositions_ant.mrk',subpath = "m2m_ernie")[0]
# Notes: 
# 1) mrk-files can contain several tms_lists. Therefore, a python list of tms-lists
#    is returned. In this example, the list has only one entry.
# 2) mrk-files can store the filename of the coil (tms_list_ant.fnamecoil)
#    which prevents the need to set it manually in simnibs
# 3) when proving the m2m-folder (here subpath = "m2m_ernie"), ant().read()
#    will check whether the positions correspond to the same T1 image
#    as used to create the headmodel with charm

# The three tms-lists above can be used in the same way to set up simulations. 
# For demonstration, the first one is used in the following:
S = sim_struct.SESSION()
S.subpath = "m2m_ernie"  # m2m-folder of the subject
S.pathfem = "tms_simu_fromimported"

tms_list_loc.fnamecoil = os.path.join("legacy_and_other", "Magstim_70mm_Fig8.ccd")  
S.add_poslist(tms_list_loc)

# Run Simulation
run_simnibs(S)


# =============================================================
# Exporting the best position of a TMS or TMS_flex optimization
# =============================================================

# set up and run a TMS or TMS_flex optimization
tms_opt = opt_struct.TMSoptimize()
tms_opt.subpath = "m2m_ernie"
tms_opt.pathfem = "tms_optimization/"
tms_opt.fnamecoil = os.path.join("legacy_and_other", "Magstim_70mm_Fig8.ccd")
tms_opt.target = [-39.7, 7.5, 65.6]
# to safe time in this demo, restrict the search to 
# only a few coil positions
tms_opt.search_radius = 8
tms_opt.search_angle = 0

opt_pos = tms_opt.run()

# Create a TMSLIST containing the optimal position
tmslist = sim_struct.TMSLIST()
tmslist.fnamecoil = tms_opt.fnamecoil
pos=sim_struct.POSITION()
pos.matsimnibs = np.squeeze(opt_pos)
tmslist.add_position(pos)

# Export positions
localite().write(tmslist, 'opt_coilpos_localite_InstrumentMarker.xml')
brainsight().write(tmslist, 'opt_coilpos_brainsight.txt')
ant().write(tmslist, 'opt_coilpos_ant.mrk', subpath=tms_opt.subpath)
