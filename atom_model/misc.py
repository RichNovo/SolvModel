import os
from atom_model.env import abstract_mapping_to_node
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt
import numpy as np
from ase.optimize.minimahopping import MinimaHopping
from ase import Atom, Atoms
import ase
import itertools
from ase.geometry.analysis import Analysis
from tblite.ase import TBLite
from ase.constraints import FixAtoms, Hookean

ohc_mix_labels = ["Propargyl alcohol",
                  "Ethanol",
                  "Penta-2,4-diynoic acid",
                  "2-Furanol",
                  "3-Furanol",
                  "Methyl propiolate",
                  "Buta-2,3-dienoic acid",
                  "2-one-3-ynebutanol",
                  "4-hydroxy-2-ynebutanal",
                  "2-hydroxy-3-ynebutanal",
                  "Cyclopropanol",
                  "Allyl alcohol",
                  "Penta-1,4-diyn-3-ol",
                  "2,4-pentadiyn-1-ol",
                  "phenol"
                  ]



def calculate_hist(nodes,nodes_xyz,fix_mask,rot,node_map):
    a_list = []

    nodes_real, nodes_xyz_real = abstract_mapping_to_node(nodes, nodes_xyz, rot, node_map)
    q = 12
    opt_path = f"./minima_hopping/200_iter/test_10_water_{str(q)}/result/minima.traj"

    opt_trajreader = []
    if os.path.exists(opt_path):
        opt_trajreader = Trajectory(opt_path,'r')

    own_path = f"./good_results/final_solv/final_v1/results/test_lp_10/example_{str(q)}_new2.traj"
    own_trajreader = []
    if os.path.exists(opt_path):
        own_trajreader = Trajectory(own_path,'r')

    a = 20.0  # lattice constant in Angstrom

    cell = [[a, 0, 0],
            [0, a, 0],
            [0, 0, a]]
    

    ncols = 3
    fig, axs = plt.subplots(nrows=15//ncols, ncols=ncols,figsize=(4*15//ncols, 4*ncols), dpi=200)
    for q in range(0,15):
        opt_path = f"./minima_hopping/200_iter/test_10_water_{str(q)}/result/minima.traj"
        orig_atoms_list = []
        for i in range(1000):
            other_path = f"./renders/rdf_test_renders/optimized_2/example_{str(q)}_{str(i)}.traj"
            if os.path.exists(other_path) is False:
                break
            traj = Trajectory(other_path,'r')[-2]
            orig_atoms_list.append(traj)
        opt_trajreader = []
        if os.path.exists(opt_path):
            opt_trajreader = Trajectory(opt_path,'r')

        #plt.clf()

        res = 100
        dist = 10.
        from_ = 11
        res = []
        for j in range(0,len(opt_trajreader)):
            res.append(opt_trajreader[j].get_potential_energy())
        res=np.array(res)[np.max(res)>res]
        offset = np.min(res)
        opt_min = np.argmin(res)
        axs[q // ncols,q%ncols].hist(res-offset,30,None,True,alpha=0.8)

        res=[]
        for j in range(0,len(orig_atoms_list)):
            res.append(orig_atoms_list[j].get_potential_energy())
        
        if len(np.argwhere(res<offset)) > 0:
            print(f"mol index: {q} traj_index: {np.argwhere(res<offset).flatten()} vs. trajindex: {opt_min}")
        #tmp = 5*np.std(res)
        #res=np.array(res)[np.logical_and((np.median(res)-tmp)<res,(np.median(res)+tmp)>res)]
        res = np.array(res)[(offset-3)<res]
        res = res[(offset+4)>res]
        axs[q // ncols,q%ncols].hist(res-offset,30,None,True,alpha=0.8)
        axs[q // ncols,q%ncols].set_title(ohc_mix_labels[q],fontsize=6)
        axs[q // ncols,q%ncols].tick_params(axis='both', which='both', labelsize=6)
        if q>11:
            axs[q // ncols,q%ncols].set_facecolor('0.8')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.4)
    plt.savefig(f'./rdf_images/minima_hopping/img_hist.png')
    
    return
    


def calculate_rdf(nodes,nodes_xyz,fix_mask,rot,node_map):
    
    a_list = []

    nodes_real, nodes_xyz_real = abstract_mapping_to_node(nodes, nodes_xyz, rot, node_map)
    q = 12
    opt_path = f"./minima_hopping/200_iter/test_10_water_{str(q)}/result/minima.traj"

    opt_trajreader = []
    if os.path.exists(opt_path):
        opt_trajreader = Trajectory(opt_path,'r')

    own_path = f"./good_results/final_solv/final_v1/results/test_lp_10/example_{str(q)}_new2.traj"
    own_trajreader = []
    if os.path.exists(opt_path):
        own_trajreader = Trajectory(own_path,'r')

    a = 20.0  # lattice constant in Angstrom

    cell = [[a, 0, 0],
            [0, a, 0],
            [0, 0, a]]

    #plt.plot(Analysis(atoms).get_rdf(10.,100,elements = [1,8])[0])
    ana = Analysis(Atoms(opt_trajreader[-1].numbers,opt_trajreader[-1].positions,cell=cell,pbc=[False, False, False]))
    #plt.plot(Analysis(atoms_o).get_rdf(10.,100,elements = [1,8])[0])


    best = False
    ncols = 3
    fig, axs = plt.subplots(nrows=15//ncols, ncols=ncols,figsize=(4*15//ncols, 4*ncols), dpi=200)
    for q in range(0,15):
        opt_path = f"./minima_hopping/200_iter/test_10_water_{str(q)}/result/minima.traj"
        orig_atoms_list = []
        for i in range(1000):
            other_path = f"./renders/rdf_test_renders/optimized_2/example_{str(q)}_{str(i)}.traj"
            if os.path.exists(other_path) is False:
                break
            traj = Trajectory(other_path,'r')[-1]
            orig_atoms_list.append(traj)
        opt_trajreader = []
        if os.path.exists(opt_path):
            opt_trajreader = Trajectory(opt_path,'r')

        if best is True:
            tmp = []
            for j in range(0,len(opt_trajreader)):
                tmp.append(opt_trajreader[j].get_potential_energy())
            argmin_opt=np.argmin(tmp)
            opt_trajreader = [opt_trajreader[argmin_opt]]
            tmp = []
            for j in range(0,len(orig_atoms_list)):
                tmp.append(orig_atoms_list[j].get_potential_energy())
            argmin_own=np.argmin(tmp)
            orig_atoms_list = [orig_atoms_list[argmin_own]]
            print(f"mol_index: {q} argmin_opt: {argmin_opt}, argmin_own: {argmin_own} energy_opt: {opt_trajreader[0].get_potential_energy()} energy_own: {orig_atoms_list[0].get_potential_energy()} better: {orig_atoms_list[0].get_potential_energy()<opt_trajreader[0].get_potential_energy()}")


        #plt.clf()
        res = 100
        dist = 10.
        from_ = 11
        sum = np.zeros(res)
        for j in range(0,len(opt_trajreader)):
            sum += Analysis(Atoms(opt_trajreader[j].numbers,opt_trajreader[j].positions,cell=cell,pbc=[False, False, False])).get_rdf(dist,res,elements = [1,8])[0]
        sum = sum/len(opt_trajreader)

        axs[q // ncols,q%ncols].plot([x/dist for x in range(0,res) ][from_:],sum[from_:])

        for j in range(0,len(orig_atoms_list)):
            sum += Analysis(ase.Atoms(orig_atoms_list[j].numbers,orig_atoms_list[j].positions,cell=cell,pbc=[False, False, False])).get_rdf(dist,res,elements = [1,8])[0]
        sum = sum/len(orig_atoms_list)
        axs[q // ncols,q%ncols].plot([x/dist for x in range(0,res) ][from_:],sum[from_:])
        axs[q // ncols,q%ncols].set_title(ohc_mix_labels[q],fontsize=6)
        axs[q // ncols,q%ncols].tick_params(axis='both', which='both', labelsize=6)
        if q>11:
            axs[q // ncols,q%ncols].set_facecolor('0.8')
        
        #plt.savefig(f'./rdf_images/minima_hopping/img_{q}.png')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.3)
    plt.savefig(f'./rdf_images/minima_hopping/img_rdf.png')

    ana.get_rdf(4.,100,elements = [1,8])

def minima_hopping_fun(nodes,nodes_xyz,fix_mask,rot,node_map,dir = "./minima_hopping/"):
    nodes_real, nodes_xyz_real = abstract_mapping_to_node(nodes, nodes_xyz, rot, node_map)
    a_list = []
    for i in range(len(nodes_real)):
        a_list.append(ase.Atom(nodes_real[i] + 1, nodes_xyz_real[i]))
    atoms = ase.Atoms(a_list)

    constraints = [FixAtoms(indices=np.argwhere(fix_mask==0.).squeeze(axis=-1)),]

    index = 0
    for i in range(len(nodes)):
        if nodes[i] in node_map:
            connected_indices = []
            for elem in node_map[nodes[i]]:
                connected_indices.append(index)
                index +=1

            comb = itertools.combinations(connected_indices, 2)
            for comb_index in comb:
                constraints.append(Hookean(a1=comb_index[0], a2=comb_index[1],
                                        rt=np.linalg.norm(nodes_xyz_real[comb_index[0]]-nodes_xyz_real[comb_index[1]], axis=-1),
                                        k=15.))
        else:
            index +=1
    atoms.set_constraint(constraints)

    # Set the calculator.
    #calc = EMT()
    #calc = TBLite(method = "GFN2-xTB")
    calc = TBLite(None, 20, method = "GFN2-xTB")
    #calc.set(verbosity = 0)
    calc.set(max_iterations = 100)
    calc.set(accuracy = 1.0)
    atoms.calc = calc

    # Instantiate and run the minima hopping algorithm.
    if os.path.exists(os.path.join(dir,'result/')) == False:
        os.makedirs(os.path.join(dir,'result/'))
    if os.path.exists(os.path.join(dir,'calc/')) == False:
        os.makedirs(os.path.join(dir,'calc/'))

    hop = MinimaHopping(atoms,
                        Ediff0=2.5,
                        T0=4000.,
                        minima_traj = os.path.join(dir,'result/',"minima.traj"),
                        logfile =  os.path.join(dir,'result/','hop.log'),
                        md_and_qn_path = os.path.join(dir,"calc/"))
    hop(totalsteps=10)