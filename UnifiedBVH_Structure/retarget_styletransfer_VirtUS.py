import os
import sys
import numpy as np
import scipy.io as io

sys.path.append('../motion')
import BVH
import Animation
from Quaternions import Quaternions
from InverseKinematics import JacobianInverseKinematics

dbroot='./'
styles = [
    'ES',
    'neutral',
    'neurotic'
    ]


bvh_files = []
    
# Iterate through all files in the folder
for file_name in os.listdir('./VirtUS_BVH'):
        # Get the absolute path of the file
        file_path = os.path.join('./VirtUS_BVH/', file_name)
        
        # Check if the path is a file (not a directory)
        if os.path.isfile(file_path):
            bvh_files.append(file_path)
    
database = bvh_files

rest, names, _ = BVH.load('./rest.bvh')

if not os.path.exists('styletransfer'): os.mkdir('styletransfer')
BVH.save('./styletransfer/rest.bvh', rest, names)

for i, filename in enumerate(database):
    
    print('%i of %i Processing %s' % (i+1, len(database), filename))
    
    stanim, stnames, ftime = BVH.load(filename)
    stanim.positions = stanim.positions / 6.0
    stanim.offsets = stanim.offsets / 6.0
    
    targets = Animation.positions_global(stanim)
    
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)
    
    anim.positions[:,0] = targets[:,0]
    anim.rotations[:,0] = stanim.rotations[:,0]

    mapping = {
         0: 0,
         1: 0,  2: 19,  3: 20,  4: 21,  5: 22,
         6: 0,  7: 15,  8: 16,  9: 17, 10: 18,
        11: 2, 12:  3, 13:  4, 15:  5, 16:  6,
        17: 4, 18: 12, 19: 13, 20: 14,
        24: 4, 25:  8, 26:  9, 27: 10}
    
    targetmap = {}
    
    for k in mapping:
        anim.rotations[:,k] = stanim.rotations[:,mapping[k]]
        targetmap[k] = targets[:,mapping[k]]
    
    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=10.0, silent=True)
    ik()
    
    BVH.save('./styletransfer/'+filename.replace('./VirtUS_BVH/',''), anim, names, ftime)
    
