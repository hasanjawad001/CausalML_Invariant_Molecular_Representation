import os
# # Navigate to the Open-Catalyst-Project directory
# os.chdir('/curdir/ocp')
# # Install the package
# !pip install -e .

import numpy as np
import ase.io
from ase.constraints import FixAtoms
from ase.build import add_adsorbate, molecule, surface
from pymatgen.ext.matproj import MPRester
from pymatgen.core.surface import generate_all_slabs, SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import torch
import pickle
import json
import pandas as pd
from pprint import pprint
from pymatgen.ext.matproj import MPRester
from pymatgen.core.surface import generate_all_slabs
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

# m = MPRester('Yct0KDbJbqMLWluZEovkwrLXh2VRHXbc')
m = MPRester('VodcKXDI4RxRtILIFIxmjg6DpM9d4iom')

with open('/curdir/datasets/ocp_reactions_info.pickle', 'rb') as f:    
    loaded_reactions = pickle.load(f)
    
print(len(loaded_reactions))

delta = 100000 
start=400000
s=start
list_se = []
while (s < start+1): ## (s < 450000):
    list_se.append((s, s+delta))
    s+=delta
    
print(list_se)

for idx, (s, e) in enumerate(list_se):
    list_rinfo = []
    for ir, r in enumerate(loaded_reactions[s:e]):
        try:
            structure = m.get_structure_by_material_id(r['bulk_mpid']) 
            miller_index = r['miller_index']
            # slabs = generate_all_slabs(structure, max_index=3, min_slab_size=10.0, min_vacuum_size=10.0)
            # for slab in slabs:
            #     if slab.miller_index == miller_index:
            #         your_slab = slab
            #         break
            slabgen = SlabGenerator(structure, miller_index, 10, 10)
            # slabs = slabgen.get_slabs()
            slabs = [slabgen.get_slab(shift=r['shift'])]
            slab = slabs[0]        
            r['slab'] = slab
            list_rinfo.append(r)
        except Exception as exp:
            pass
        
    save_path = os.path.join('/curdir/datasets/slabs', f'list_rinfo_{s}to{e}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(list_rinfo, f)
    print(f"done: {s}to{e}")