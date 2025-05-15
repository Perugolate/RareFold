import json
import os
import warnings
import pathlib
import pickle
import random
import sys
sys.path.insert(0, '/proj/berzelius-2023-267/users/x_patbr/software/rare_fold/src/net/')
sys.path.insert(0, '/home/bryant/software/rare_fold/src/net/')
import time
from typing import Dict, Optional
from typing import NamedTuple
import haiku as hk
import jax
import jax.numpy as jnp
import optax
#Silence tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.config.set_visible_devices([], 'GPU')

import argparse
import pandas as pd
import numpy as np
from collections import Counter
from scipy.special import softmax
import copy
from ast import literal_eval

import pdb


#AlphaFold imports - now RareFold
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.model import data
from alphafold.model import config
from alphafold.model import features
from alphafold.model import modules

#JAX will preallocate 90% of currently-available GPU memory when the first JAX operation is run.
#This prevents this
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


parser = argparse.ArgumentParser(description = """Design a binder using trained weights.""")


parser.add_argument('--predict_id', nargs=1, type= str, default=sys.stdin, help = 'Id to predict.')
parser.add_argument('--MSA_feats', nargs=1, type= str, default=sys.stdin, help = 'Id to predict.')
parser.add_argument('--num_recycles', nargs=1, type= int, default=sys.stdin, help = 'Number of recycles.')
parser.add_argument('--binder_length', nargs=1, type= int, default=sys.stdin, help = 'Length of binder.')
parser.add_argument('--num_iterations', nargs=1, type= int, default=sys.stdin, help = 'Number of iterations to run.')
parser.add_argument('--resample_every_n', nargs=1, type= int, default=sys.stdin, help = 'How often to resample the MSA - avoids local minima.')
parser.add_argument('--batch_size', nargs=1, type= int, default=sys.stdin, help = 'Batch size (will run design threads in parallel).')
parser.add_argument('--params', nargs=1, type= str, default=sys.stdin, help = 'Params to use.')
parser.add_argument('--rare_AAs', nargs=1, type= str, default=sys.stdin, help = 'List of rare amino acids to use in the design.')
parser.add_argument('--cyclic_offset', nargs=1, type= str, default=sys.stdin, help = 'Use a cyclic offset for the binder (True) or not (False).')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

##############FUNCTIONS##############
##########INPUT DATA#########
def process_features(raw_features, config, random_seed):
    """Processes features to prepare for feeding them into the model.

    Args:
    raw_features: The output of the data pipeline either as a dict of NumPy
      arrays or as a tf.train.Example.
    random_seed: The random seed to use when processing the features.

    Returns:
    A dict of NumPy feature arrays suitable for feeding into the model.
    """
    return features.np_example_to_features(np_example=raw_features,
                                            config=config,
                                            random_seed=random_seed)



def process_input_feats(new_feature_dict, config):
    """
    Load all input feats.
    """


    #Number of possible amino acids
    num_AAs = len(residue_constants.restype_name_to_atom14_names.keys())
    #Max number of atoms per amino acid in the dense representation
    num_dense_atom_max = len(residue_constants.restype_name_to_atom14_names['ALA'])
    #Process the features on CPU (sample MSA)
    #This also creates mappings for the atoms: 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists'
    new_feature_dict['aatype'] =  np.eye(num_AAs)[new_feature_dict['int_seq']]
    processed_feature_dict = process_features(new_feature_dict, config, np.random.choice(sys.maxsize))

    #Cyclic
    if config.model.embeddings_and_evoformer.cyclic_offset==True:
        pos = new_feature_dict['residue_index']
        cyclic_offset_array = pos[:, None] - pos[None, :]
        binder_cyclic_offset_array = new_feature_dict['binder_cyclic_offset_array']
        cyclic_offset_array[-len(binder_cyclic_offset_array):,-len(binder_cyclic_offset_array):]=binder_cyclic_offset_array
        new_feature_dict['cyclic_offset'] = cyclic_offset_array



    #Arrange feats
    batch_ex = copy.deepcopy(new_feature_dict)

    #If Rare amino acids in the receptor - this has to be specified here
    #batch_ex['aatype'] = rare_feats['onehot_seq'] #Use the sequence from the structure here - RARE!!!
    batch_ex['aatype'] = new_feature_dict['int_seq']
    batch_ex['seq_mask'] = processed_feature_dict['seq_mask']
    batch_ex['msa_mask'] = processed_feature_dict['msa_mask']
    batch_ex['residx_atom14_to_atom37'] = processed_feature_dict['residx_atom14_to_atom37']
    batch_ex['residx_atom37_to_atom14'] = processed_feature_dict['residx_atom37_to_atom14']
    batch_ex['atom37_atom_exists'] = processed_feature_dict['atom37_atom_exists']
    batch_ex['extra_msa'] = processed_feature_dict['extra_msa']
    batch_ex['extra_msa_mask'] = processed_feature_dict['extra_msa_mask']
    batch_ex['bert_mask'] = processed_feature_dict['bert_mask']
    batch_ex['true_msa'] = processed_feature_dict['true_msa']
    batch_ex['extra_has_deletion'] = processed_feature_dict['extra_has_deletion']
    batch_ex['extra_deletion_value'] = processed_feature_dict['extra_deletion_value']
    batch_ex['msa_feat'] = processed_feature_dict['msa_feat']

    #Target feats have to be updated with the onehot_seq from the structure to include the modified amino acids
    batch_ex['target_feat'] = np.eye(num_AAs)[new_feature_dict['int_seq']]
    batch_ex['atom14_atom_exists'] = processed_feature_dict['atom14_atom_exists']
    batch_ex['residue_index'] = processed_feature_dict['residue_index']

    return batch_ex


def init_features(feature_dict, onehot_binder_seq, binder_length, config):
    """Update the features to include the binder sequence

    #From MSA feats
    'aatype',
    'between_segment_residues',
    'domain_name',
    'residue_index',
    'seq_length',
    'sequence',
    'deletion_matrix_int',
    'msa',
    'num_alignments'
    """

    #Save
    new_feature_dict = {}

    #Add peptide feats to feature dict
    #aatype
    new_feature_dict['int_seq'] = np.concatenate((np.argmax(feature_dict['aatype'],axis=1), np.array(onehot_binder_seq)),axis=0)
    #between_segment_residues
    new_feature_dict['between_segment_residues'] = np.concatenate((feature_dict['between_segment_residues'],np.zeros((binder_length), dtype=np.int32)),axis=0)
    #residue_index
    new_feature_dict['residue_index'] = np.concatenate((feature_dict['residue_index'],np.array(range(binder_length), dtype=np.int32)+feature_dict['residue_index'][-1]+201), axis=0)
    #seq_length
    new_feature_dict['seq_length'] = np.array([new_feature_dict['int_seq'].shape[0]] * new_feature_dict['int_seq'].shape[0], dtype=np.int32)

    #Merge MSA features
    #deletion_matrix_int
    new_feature_dict['deletion_matrix_int']=np.concatenate((feature_dict['deletion_matrix_int'],
                                            np.zeros((feature_dict['deletion_matrix_int'].shape[0],binder_length))), axis=1)
    #msa
    peptide_msa = np.zeros((feature_dict['msa'].shape[0],binder_length),dtype=int)
    peptide_msa[:,:] = 21
    #Assign first seq - need to have X instead of mod AAs
    """
    HHBLITS_AA_TO_ID = {'A': 0,'B': 2,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'J': 20,'K': 8,'L': 9,'M': 10,'N': 11,
                        'O': 20,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'U': 1,'V': 17,'W': 18,'X': 20,'Y': 19,'Z': 3,'-': 21,}
    """
    x = copy.deepcopy(np.array(onehot_binder_seq))
    x[x>19]=20
    peptide_msa[0,:] = x

    new_feature_dict['msa']=np.concatenate((feature_dict['msa'], peptide_msa), axis=1)

    #num_alignments
    new_feature_dict['num_alignments']=np.concatenate((feature_dict['num_alignments'], feature_dict['num_alignments'][:len(onehot_binder_seq)]), axis=0)

    #Process
    if config.model.embeddings_and_evoformer.cyclic_offset==True:
        new_feature_dict['binder_cyclic_offset_array'] = copy.deepcopy(feature_dict['binder_cyclic_offset_array'])

    new_feature_dict = process_input_feats(new_feature_dict, config)


    return new_feature_dict



##########MODEL and DESIGN#########
def initialize_weights(binder_length, rare_AAs, batch_size):
    '''Initialize sequence probabilities
    '''

    all_AAs = np.array([*residue_constants.restype_name_to_atom14_names.keys()])
    #Select the ones that are in design AAs
    selected_AA_index = [x for x in range(20)]
    for raa in rare_AAs:
        selected_AA_index.append(np.argwhere(all_AAs==raa)[0][0])

    selected_AA_index = np.sort(np.unique(selected_AA_index))
    num_design_AAs = len(selected_AA_index)

    binder_seqs, onehot_binder_seqs = [], []
    for i in range(batch_size):
        weights = np.random.gumbel(0,1,(binder_length, num_design_AAs))
        weights = np.array([np.exp(weights[i])/np.sum(np.exp(weights[i])) for i in range(len(weights))])

        #Get the peptide sequence
        onehot_binder_seqs.append([selected_AA_index[x] for x in np.argmax(weights,axis=1)])
        binder_seqs.append('-'.join(all_AAs[onehot_binder_seqs[-1]]))

    return binder_seqs, onehot_binder_seqs, all_AAs, selected_AA_index


def mutate_sequence(onehot_binder_seq, searched_seqs, all_AA_triplets, selected_AA_index):
    '''Mutate the amino acid sequence randomly
    '''

    seqlen = len(onehot_binder_seq)
    searched_seqs = [list(x) for x in searched_seqs]
    #Mutate seq
    seeds = [onehot_binder_seq]
    #Go through a shuffled version of the positions and aas
    for seed in seeds:
        #Get position to mutate
        for pi in np.random.choice(np.arange(seqlen),seqlen, replace=False):
            #Get restype
            for aa in np.random.choice(selected_AA_index,len(selected_AA_index), replace=False):
                new_seq = copy.deepcopy(seed)
                new_seq = new_seq[:pi]+[aa]+new_seq[pi+1:]
                if new_seq in searched_seqs:
                    continue
                else:
                    return new_seq, '-'.join(all_AA_triplets[new_seq])

        seeds.append(new_seq)

def get_atom_mapping_per_restype():
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    restype_atom14_mask = []

    for rt in [*residue_constants.restype_name_to_atom14_names.keys()]:
        atom_names = residue_constants.restype_name_to_atom14_names[rt]

        restype_atom14_to_atom37.append([
            (residue_constants.atom_order[name] if name else 0)
            for name in atom_names
        ])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([
            (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
            for name in residue_constants.atom_types
        ])

        restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

    # create the corresponding mask
    #####MOD for RARE AAs#####
    restype_atom37_mask = np.zeros([residue_constants.restype_num, len(residue_constants.atom_order.keys())], dtype=np.float32)

    for restype_name in residue_constants.residue_atoms:
        atom_names = residue_constants.residue_atoms[restype_name]
        restype = residue_constants.resname_order[restype_name] #Get index for resname, MOD for RARE!!!
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1


    restype_atom_mappings = {'restype_atom14_to_atom37':restype_atom14_to_atom37,
                             'restype_atom37_to_atom14':restype_atom37_to_atom14,
                             'restype_atom14_mask':restype_atom14_mask,
                             'restype_atom37_mask':restype_atom37_mask
                            }

    return restype_atom_mappings, len(residue_constants.restype_name_to_atom14_names.keys())



def update_peptide_batch_feats(batch, int_binder_seqs, binder_length, num_AAs, restype_atom_mappings):
    """Update only the peptide batch feats that affect the prediction
    int_seq: batch_size,1,L
    residx_atom14_to_atom37: batch_size,1,L,25
    esidx_atom37_to_atom14
    atom37_atom_exists
    target_feat
    atom14_atom_exists
    """

    target_feat = np.array([np.eye(num_AAs)[x] for x in int_binder_seqs])
    batch['target_feat'][:,:,-binder_length:,:] = np.expand_dims(target_feat, axis=1)
    batch['int_seq'][:,:,-binder_length:] = np.expand_dims(int_binder_seqs, axis=1)
    batch['aatype'][:,:,-binder_length:] = np.expand_dims(int_binder_seqs, axis=1)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein)
    #aatype = np.array([np.eye(num_AAs)[x] for x in int_binder_seqs])
    residx_atom14_to_atom37 = tf.gather(restype_atom_mappings['restype_atom14_to_atom37'], int_binder_seqs)
    residx_atom14_mask = tf.gather(restype_atom_mappings['restype_atom14_mask'], int_binder_seqs)

    #Update batch
    batch['residx_atom14_to_atom37'][:,:,-binder_length:,:] = np.expand_dims(residx_atom14_to_atom37, axis=1)
    batch['atom14_atom_exists'][:,:,-binder_length:,:] = np.expand_dims(residx_atom14_mask, axis=1)

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = tf.gather(restype_atom_mappings['restype_atom37_to_atom14'], int_binder_seqs)
    batch['residx_atom37_to_atom14'][:,:,-binder_length:,:] = np.expand_dims(residx_atom37_to_atom14, axis=1)

    residx_atom37_mask = tf.gather(restype_atom_mappings['restype_atom37_mask'], int_binder_seqs)
    batch['atom37_atom_exists'][:,:,-binder_length:,:] = np.expand_dims(residx_atom37_mask, axis=1)

    return batch

def get_loss(prediction_result, bi, binder_length):
    '''Predict and calculate loss
    '''

    #Calculate loss
    #Loss features
    # Get the pLDDT confidence metric.
    #Define the plDDT bins
    bin_width = 1.0 / 50
    bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)

    #Add the predicted LDDT in the b-factor column.
    plddt_per_pos = jnp.sum(jax.nn.softmax(prediction_result['predicted_lddt']['logits'][bi]) * bin_centers[None, :], axis=-1)

    final_atom_positions = prediction_result['structure_module']['final_atom_positions'][bi]
    final_atom_mask = prediction_result['structure_module']['final_atom_mask'][bi]

    #Go through the positions and only add where the mask is >0.5
    extracted_coords = []
    extracted_resnos = []
    ri=1
    for i in range(len(final_atom_positions)):
        pos_i = final_atom_positions[i]
        mask_i = final_atom_mask[i]
        sel_inds = np.argwhere(mask_i>0.5)[:,0]
        extracted_coords.extend([*pos_i[sel_inds]])
        extracted_resnos.extend([ri]*len(sel_inds))
        ri+=1


    #Divide by receptor/peptide
    u_resnos = np.unique(extracted_resnos)
    peptide_resnos = u_resnos[-binder_length:]
    receptor_inds = np.argwhere(np.array(extracted_resnos)<peptide_resnos[0])[:,0]
    peptide_inds = np.argwhere(np.array(extracted_resnos)>=peptide_resnos[0])[:,0]

    extracted_coords = np.array(extracted_coords)
    receptor_coords = extracted_coords[receptor_inds]
    peptide_coords = extracted_coords[peptide_inds]


    #Calc 2-norm - distance between peptide and interface
    mat = np.concatenate([peptide_coords, receptor_coords],axis=0)
    dists = np.sqrt(np.sum((mat[:,None] - mat[None,:])**2, axis=-1))
    l1 = len(peptide_coords)
    #Get interface
    contact_dists = dists[:l1,l1:] ##first dimension = peptide, second = receptor
    #Get clashes
    #Inter
    inter_clashes = np.argwhere(contact_dists < 1.5)
    inter_clash_frac = inter_clashes.shape[0]/(1e-7+contact_dists.shape[0])
    #Intra
    binder_intra_dists = dists[:l1,:l1]
    intra_clashes = np.argwhere(binder_intra_dists < 1)
    intra_clash_frac = intra_clashes.shape[0]/(1e-7+l1**2)

    #Get the closest atom-atom distances across the receptor interface residues.
    closest_dists_peptide = contact_dists[np.arange(contact_dists.shape[0]),np.argmin(contact_dists,axis=1)]

    #Get the binder plDDT
    binder_plDDT = plddt_per_pos[-binder_length:]

    return closest_dists_peptide.mean(), binder_plDDT.mean()*100, inter_clash_frac, intra_clash_frac/10


def design_binder(config,
                predict_id,
                MSA_feats,
                num_recycles=3,
                binder_length=10,
                num_iterations=1000,
                resample_every_n=100,
                batch_size=1,
                params=None,
                rare_AAs=['MSE'],
                outdir=None):
    """Design a binder
    """

    #Initialize weights - these are the amino acid probabilities
    #Also returns the peptide_sequence corresponding to the weights
    binder_seqs, int_binder_seqs, all_AA_triplets, selected_AA_index = initialize_weights(binder_length, rare_AAs, batch_size)

    if config.model.embeddings_and_evoformer.cyclic_offset==True:
        cyclic_offset_array = np.zeros((binder_length, binder_length))
        cyc_row = np.arange(0,-binder_length,-1)
        pc = int(np.round(binder_length/2)) #Get centre
        cyc_row[pc+1:]=np.arange(len(cyc_row[pc+1:]),0,-1)
        for i in range(len(cyclic_offset_array)):
            cyclic_offset_array[i]=np.roll(cyc_row,i)
        MSA_feats['binder_cyclic_offset_array']=cyclic_offset_array

    #Define the forward function
    def _forward_fn(batch):
        '''Define the forward function - has to be a function for JAX
        '''
        model = modules.AlphaFold(config.model)

        return model(batch,
                    is_training=False,
                    compute_loss=False,
                    ensemble_representations=False,
                    return_representations=True)

    #The forward function is here transformed to apply and init functions which
    #can be called during training and initialisation (JAX needs functions)
    forward = hk.transform(_forward_fn)
    #Function to vmap - this is usually wrapped with functools
    #This causes communication errors btw my processes in HPC envs, however
    vmap_apply_fwd = jax.vmap(forward.apply, (None,None,0)) #None over params and rng, but 0 over batch

    #Get a random key
    rng = jax.random.PRNGKey(42)

    #Load params (need to do this here - need to enable GPU through jax first)
    params = np.load(params, allow_pickle=True)


    ####Run the directed evolution####
    sequence_scores = {'iteration':[],
                        'if_dist_binder':[],
                        'plddt':[],
                        'inter_clash_frac':[],
                        'intra_clash_frac':[],
                        'loss':[],
                        'sequence':[],
                        'int_seq':[],
                        }

    #check if previous run exists
    if os.path.exists(outdir+'metrics.csv'):
        print('Run exists, continuing...')
        score_df = pd.read_csv(outdir+'metrics.csv')
        for col in score_df.columns:
            if col!='iteration':
                sequence_scores[col] = [literal_eval(x) for x in score_df[col].values]
            else:
                sequence_scores[col] = [*score_df[col].values]

        #Reset starting point to min
        best_inds = np.argmin(sequence_scores['loss'],axis=0)
        int_binder_seqs = []
        for i in range(len(best_inds)):
            int_binder_seqs.append(sequence_scores['int_seq'][best_inds[i]][i])

        #Load batch
        init_feature_dicts = [init_features(MSA_feats, int_binder_seqs[i], binder_length, config) for i in range(batch_size)]
        batch = {}
        for key in init_feature_dicts[0]:
            batch[key] = np.array([init_feature_dicts[x][key] for x in range(batch_size)])
            batch[key] = np.reshape(batch[key], (batch_size, 1, *batch[key].shape[1:]))

        batch['num_iter_recycling'] = np.zeros((batch_size, 1,))
        batch['num_iter_recycling'][:] = num_recycles


    else:
        print('No previous run found. Starting new...')
        #Make feature dicts
        init_feature_dicts = [init_features(MSA_feats, int_binder_seqs[i], binder_length, config) for i in range(batch_size)]
        batch = {}
        for key in init_feature_dicts[0]:
            batch[key] = np.array([init_feature_dicts[x][key] for x in range(batch_size)])
            batch[key] = np.reshape(batch[key], (batch_size, 1, *batch[key].shape[1:]))

        batch['num_iter_recycling'] = np.zeros((batch_size, 1,))
        batch['num_iter_recycling'][:] = num_recycles


        print('Predicting init...')
        t0 = time.time()
        prediction_result = vmap_apply_fwd(params, rng, batch)
        print('Init took',time.time()-t0,'s')
        #Save all init
        t0 = time.time()
        [save_structure(batch, prediction_result, i, 'init', outdir) for i in range(batch_size)]
        print('Saving init took',time.time()-t0,'s')

        #Loss
        iter_loss_metrics = [get_loss(prediction_result, i, binder_length) for i in range(batch_size)]

        if_dist_binder = np.array([x[0] for x in iter_loss_metrics])
        plddt = np.array([x[1] for x in iter_loss_metrics])
        inter_clash_fracs = np.array([x[2] for x in iter_loss_metrics])
        intra_clash_fracs = np.array([x[3] for x in iter_loss_metrics])
        #Add to scores
        sequence_scores['iteration'].append('init')
        sequence_scores['if_dist_binder'].append([*if_dist_binder])
        sequence_scores['plddt'].append([*plddt])
        sequence_scores['inter_clash_frac'].append([*inter_clash_fracs])
        sequence_scores['intra_clash_frac'].append([*intra_clash_fracs])
        loss = if_dist_binder*1/plddt+inter_clash_fracs+intra_clash_fracs
        sequence_scores['loss'].append([*loss])
        sequence_scores['sequence'].append(binder_seqs)
        sequence_scores['int_seq'].append(int_binder_seqs)
        #Save
        score_df = pd.DataFrame.from_dict(sequence_scores)
        score_df.to_csv(outdir+'metrics.csv', index=None)


    #Get restype atom mappings - will be used to update the peptide feats each iteration
    #These are index-based, following the same order as in resiue_constants (used for the int_seq)
    restype_atom_mappings, num_AAs = get_atom_mapping_per_restype()


    #Iterate - mutate - score - repeat
    for niter in range(len(sequence_scores['iteration']), num_iterations+1):
        #Can't prefetch - dependent on the previous iter
        #Mutate sequence
        t_0 = time.time()
        mut_seqs = [mutate_sequence(int_binder_seqs[i], np.array(sequence_scores['int_seq'])[:,i], all_AA_triplets, selected_AA_index) for i in range(batch_size)]
        int_binder_seqs = [x[0] for x in mut_seqs]
        binder_seqs = [x[1] for x in mut_seqs]
        print('Mutating sequences took', time.time() - t_0,'s')

        t_0 = time.time()
        if niter%resample_every_n==0:
            #Reload batch
            init_feature_dicts = [init_features(MSA_feats, int_binder_seqs[i], binder_length, config) for i in range(batch_size)]
            batch = {}
            for key in init_feature_dicts[0]:
                batch[key] = np.array([init_feature_dicts[x][key] for x in range(batch_size)])
                batch[key] = np.reshape(batch[key], (batch_size, 1, *batch[key].shape[1:]))

            batch['num_iter_recycling'] = np.zeros((batch_size, 1,))
            batch['num_iter_recycling'][:] = num_recycles
        else:
            #Update feats with binder seq
            t_0 = time.time()
            batch = update_peptide_batch_feats(batch, np.array(int_binder_seqs), binder_length, num_AAs, restype_atom_mappings)
        print('Making new feats took', time.time() - t_0,'s')

        #Predict - vmap over batch dim
        t_0 = time.time()
        prediction_result = vmap_apply_fwd(params, rng, batch)
        print('Prediction took', time.time() - t_0,'s')

        #Get loss
        t_0 = time.time()
        iter_loss_metrics = [get_loss(prediction_result, i, binder_length) for i in range(batch_size)]
        print('Loss calcs took', time.time() - t_0,'s')

        t_0 = time.time()
        if_dist_binder = np.array([x[0] for x in iter_loss_metrics])
        plddt = np.array([x[1] for x in iter_loss_metrics])
        inter_clash_fracs = np.array([x[2] for x in iter_loss_metrics])
        intra_clash_fracs = np.array([x[3] for x in iter_loss_metrics])

        #Add to scores
        sequence_scores['iteration'].append(str(niter))
        sequence_scores['if_dist_binder'].append([*if_dist_binder])
        sequence_scores['plddt'].append([*plddt])
        sequence_scores['inter_clash_frac'].append([*inter_clash_fracs])
        sequence_scores['intra_clash_frac'].append([*intra_clash_fracs])
        loss = if_dist_binder*1/plddt+inter_clash_fracs+intra_clash_fracs
        sequence_scores['loss'].append([*loss])
        sequence_scores['sequence'].append(binder_seqs)
        sequence_scores['int_seq'].append(int_binder_seqs)
        #Save
        score_df = pd.DataFrame.from_dict(sequence_scores)
        score_df.to_csv(outdir+'metrics.csv', index=None)

        print(niter, np.round(plddt[0],2), np.round(if_dist_binder[0], 2), np.round(inter_clash_fracs[0], 2), np.round(intra_clash_fracs[0], 2), np.round(loss[0], 3), binder_seqs[0])
        #Reset starting point to min
        best_inds = np.argmin(sequence_scores['loss'],axis=0)
        int_binder_seqs = []
        for i in range(len(best_inds)):
            int_binder_seqs.append(sequence_scores['int_seq'][best_inds[i]][i])
            #Check if improvement --> save
            if best_inds[i]==len(sequence_scores['loss'])-1:
                #Save structure
                save_structure(batch, prediction_result, i, 'unrelaxed_'+str(niter), outdir)


        print('Adding metrics took', time.time() - t_0,'s')

def save_structure(batch, prediction_result, i, pred_id, outdir):
    """Save prediction

    save_feats = {'aatype':batch['aatype'][0][0], 'residue_index':batch['residue_index'][0][0]}
    result = {'predicted_lddt':aux['predicted_lddt'],
            'structure_module':{'final_atom_positions':aux['structure_module']['final_atom_positions'][0],
            'final_atom_mask': aux['structure_module']['final_atom_mask'][0]
            }}
    save_structure(save_feats, result, step_num, outdir)

    """
    #Define the plDDT bins
    bin_width = 1.0 / 50
    bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)

    #for i in range(len(batch['aatype'])):
    #Save structure
    save_feats = {'aatype':batch['aatype'][i], 'residue_index':batch['residue_index'][i]}
    result = {'predicted_lddt':prediction_result['predicted_lddt']['logits'][i],
            'structure_module':{'final_atom_positions':prediction_result['structure_module']['final_atom_positions'][i],
            'final_atom_mask': prediction_result['structure_module']['final_atom_mask'][i]
            }}
    # Add the predicted LDDT in the b-factor column.
    plddt_per_pos = jnp.sum(jax.nn.softmax(result['predicted_lddt']) * bin_centers[None, :], axis=-1)
    plddt_b_factors = np.repeat(plddt_per_pos[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(features=save_feats, result=result,  b_factors=plddt_b_factors)
    unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(outdir+'/', pred_id+'_'+str(i)+'.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdb)



##################MAIN#######################

#Parse args
args = parser.parse_args()
predict_id = args.predict_id[0]
MSA_feats = np.load(args.MSA_feats[0], allow_pickle=True)
num_recycles = args.num_recycles[0]
binder_length = args.binder_length[0]
num_iterations = args.num_iterations[0]
resample_every_n = args.resample_every_n[0]
batch_size = args.batch_size[0]
params = args.params[0]
rare_AAs = args.rare_AAs[0].split(',')
cyclic_offset = args.cyclic_offset[0]
if cyclic_offset=='True':
    cyclic_offset=True
else:
    cyclic_offset=None
outdir = args.outdir[0]


#Update config
config.CONFIG.model.embeddings_and_evoformer['cyclic_offset'] = cyclic_offset

#Predict
design_binder(config.CONFIG,
            predict_id,
            MSA_feats,
            num_recycles=num_recycles,
            binder_length=binder_length,
            num_iterations=num_iterations,
            resample_every_n=resample_every_n,
            batch_size=batch_size,
            params=params,
            rare_AAs=rare_AAs,
            outdir=outdir)
