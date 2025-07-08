import json
import os
import warnings
import pathlib
import pickle
import random
import sys
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

#Custom imports from rarefold
from rarefold.common import protein
from rarefold.common import residue_constants
from rarefold.model import data
from rarefold.model import config
from rarefold.model import features
from rarefold.model import modules

import argparse
import pandas as pd
import numpy as np
from scipy.special import softmax
import copy
import re
import pdb


parser = argparse.ArgumentParser(description = """Predict a protein structure with noncanonical amino acids using trained weights.""")

parser.add_argument('--predict_id', nargs=1, type= str, default=sys.stdin, help = 'Id for prediction.')
parser.add_argument('--MSA_feats', nargs=1, type= str, default=sys.stdin, help = 'MSA features.')
parser.add_argument('--fasta', nargs=1, type= str, default=sys.stdin, help = 'Path to fasta for protein (containing threeletter codes for NCAAs).')
parser.add_argument('--num_recycles', nargs=1, type= int, default=sys.stdin, help = 'Number of recycles.')
parser.add_argument('--params', nargs=1, type= str, default=sys.stdin, help = 'Params to use.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

##############FUNCTIONS##############
##########INPUT DATA#########
def read_fasta(filename):
    """Read a fasta sequence with NCAAs
    """

    with open(filename, 'r') as file:
        for line in file:
            if line[0]=='>':
                continue
            else:
                line = line.rstrip()
                return line



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

def make_features(feature_dict, int_protein_seq, config):
    """Make the features

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

    #Add int_seq
    feature_dict['int_seq'] = np.array(int_protein_seq)

    #Number of possible amino acids
    num_AAs = len(residue_constants.restype_name_to_atom14_names.keys())
    #Max number of atoms per amino acid in the dense representation
    num_dense_atom_max = len(residue_constants.restype_name_to_atom14_names['ALA'])
    #Onehot
    feature_dict['aatype'] = np.eye(num_AAs)[feature_dict['int_seq']]
    #Process the features on CPU (sample MSA)
    #This also creates mappings for the atoms: 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists'
    processed_feature_dict = process_features(feature_dict, config, np.random.choice(sys.maxsize))

    #Arrange feats
    batch_ex = copy.deepcopy(feature_dict)
    batch_ex['aatype'] = feature_dict['int_seq']
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
    batch_ex['target_feat'] =  np.eye(num_AAs)[feature_dict['int_seq']]
    batch_ex['atom14_atom_exists'] = processed_feature_dict['atom14_atom_exists']
    batch_ex['residue_index'] = processed_feature_dict['residue_index']

    return batch_ex


    return new_feature_dict


##########MODEL and DESIGN#########
def get_int_seq(int_protein_seq, protein_seq):
    """
    #Map the protein_seq to int representation
    """

    all_AAs = np.array([*residue_constants.restype_name_to_atom14_names.keys()])

    ps = protein_seq.split('-') #Split to get NCAAs
    psi = 0 #Keep track of split index
    for i in np.argwhere(np.array(int_protein_seq)==20)[:,0]:
        #Replace
        if i>0:
            AA = ps[psi+1]
        else:
            AA = ps[psi]
        try:
            int_protein_seq[i] = np.argwhere(all_AAs==AA)[0][0]
        except:
            print('Could not map AA', AA)
            print('The available AAs are', all_AAs)
            sys.exit()

        psi+=1

    mapped_protein_seq = '-'.join([all_AAs[x] for x in int_protein_seq])
    return int_protein_seq, mapped_protein_seq

def predict(config,
            predict_id,
            MSA_feats,
            protein_seq,
            num_recycles=3,
            params=None,
            outdir=None):
    """Predict a protein-peptide complex where the peptide has
    non-nanonical amino acids
    """



    int_protein_seq = [*np.argmax(MSA_feats['aatype'],axis=1)]
    int_protein_seq, mapped_protein_seq = get_int_seq(int_protein_seq, protein_seq)

    print('Using protein sequence (threeletter code)', mapped_protein_seq)

    #Define the forward function
    def _forward_fn(batch):
        '''Define the forward function - has to be a function for JAX
        '''
        model = modules.RareFold(config.model)

        return model(batch,
                    is_training=False,
                    compute_loss=False,
                    ensemble_representations=False,
                    return_representations=True)

    #The forward function is here transformed to apply and init functions which
    #can be called during training and initialisation (JAX needs functions)
    forward = hk.transform(_forward_fn)
    apply_fwd = forward.apply
    #Get a random key
    rng = jax.random.PRNGKey(42)

    #Load params (need to do this here - need to enable GPU through jax first)
    params = np.load(params , allow_pickle=True)
    #Fix naming - tha params are saved using an old naming (alphafold)
    new_params = {}
    for key in params:
        new_key = re.sub('alphafold', 'rarefold', key)
        new_params[new_key] = params[key]
    params = new_params

    #Get a feature dict that includes the peptide
    feature_dict = make_features(MSA_feats, int_protein_seq, config)
    batch = {}
    for key in feature_dict:
        batch[key] = np.reshape(feature_dict[key], (1, *feature_dict[key].shape))
    batch['num_iter_recycling'] = [num_recycles]

    print('Predicting...')
    t0=time.time()
    prediction_result = apply_fwd(params, rng, batch)
    print('Prediction took', time.time()-t0, 's.')
    #Save structure
    save_feats = {'aatype':batch['aatype'], 'residue_index':batch['residue_index']}
    result = {'predicted_lddt':prediction_result['predicted_lddt'],
        'structure_module':{'final_atom_positions':prediction_result['structure_module']['final_atom_positions'],
        'final_atom_mask': prediction_result['structure_module']['final_atom_mask']
        }}
    save_structure(save_feats, result, predict_id, outdir)



def save_structure(save_feats, result, id, outdir):
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

    # Add the predicted LDDT in the b-factor column.
    plddt_per_pos = jnp.sum(jax.nn.softmax(result['predicted_lddt']['logits']) * bin_centers[None, :], axis=-1)
    plddt_b_factors = np.repeat(plddt_per_pos[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(features=save_feats, result=result,  b_factors=plddt_b_factors)
    unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(outdir+'/', id+'_pred.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdb)



##################MAIN#######################

#Parse args
args = parser.parse_args()
predict_id = args.predict_id[0]
MSA_feats = np.load(args.MSA_feats[0], allow_pickle=True)
protein_seq = read_fasta(args.fasta[0])
num_recycles = args.num_recycles[0]
params = args.params[0]
outdir = args.outdir[0]

#Set cyclic offset to None - used for cyclic peptide binder design
config.CONFIG.model.embeddings_and_evoformer['cyclic_offset'] = None

#Predict
predict(config.CONFIG,
            predict_id,
            MSA_feats,
            protein_seq,
            num_recycles=num_recycles,
            params=params,
            outdir=outdir)
