import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import RDKFingerprint, AllChem
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import faiss
from typing import List,Tuple
from rdkit import RDLogger     
from faiss import write_index, read_index


RDLogger.DisableLog('rdApp.*')   

def predict_rxn(rxn, 
z                n_expert = 3, 
                verbose = False,
                beam_size = 20,
                beam_group = 2,
                return_sequence = 20,
                diversity_penalty = 0.1,
                run_prediction = True,
                show_expert_reagent_info = False,
                show_reference = True,
                n_references = 1,
               ):
    
    if type(rxn) == type([]):
        clean_reactions = rxn
    elif type(rxn) == type(''):
        clean_reactions = [rxn]
    
    fp_size = 1024


    test_features = []
    with torch.no_grad():

        for i in range(len(clean_reactions)):
            rxn = clean_reactions[i]
            

            rxn_fp = create_rxn_Mix_FP(rxn, rxnfpsize=fp_size, pfpsize=fp_size, useChirality=True) 
   
            rxn_fp = np.concatenate((rxn_fp[1],rxn_fp[2],rxn_fp[0]), axis = -1) # reactant, diff, product

            # This considers we are only predicting a few reactions. But for other purposes, it can be optimized for batch processing
            feat = model.get_embeddings(torch.from_numpy(rxn_fp).view(1,-1).float().to(device)) # Using pre-trained embeddings

            test_features.append(feat.cpu().numpy())

    test_features = np.array(test_features).reshape((len(clean_reactions),-1))
    
        
    
    distance, cell_ids = index.quantizer.search(test_features, k=n_expert)
    
    
    if verbose:
        print('You have requested {n_expert} experts and here is the domain of knowledge of these experts:'.format(n_expert = n_expert))
        print()
        # Print overview before predictions if verbose
        for i,rxn in enumerate(clean_reactions):
            print('-'*50)
            print('Rxn.' + str(i+1) +':',rxn)
            
            show_rxn(rxn)
            
            print()
            
            for k in range(n_expert):
                print()
                print('*'*50)
               
                print('Top-' + str(k+1))
                print('Expert ID:' , cell_ids[i][k])
                print('Expert Centroid Distances' ,f"{distance[i][k]:.2f}")
                sub_df = df[df['quadrant'].values == cell_ids[i][k]]
                displaydict = dict(sub_df['namerxndef'].value_counts())
                print('Total Rxns:', len(sub_df))
                for namedef, counts in displaydict.items():
                    percentage = 100*counts/len(sub_df)
                    
                    print(namedef, f"{percentage:.2f}%")
                
                if show_expert_reagent_info:
                    # Not recommended to print everything
                    # this section will display the reagent statistics if chosen from the function
                    displaydict = dict(sub_df['agent_name'].value_counts()) 
                    for namedef, counts in displaydict.items():
                        percentage = 100*counts/len(sub_df)
                        print(namedef, f"{percentage:.2f}%")
                
                
    if not(run_prediction):
        
        return
           
   
    print('Expert Predictions:')
    for i,rxn in enumerate(clean_reactions):
        
        print('>'*50)
        print('Predicting Reaction '+str(i+1) +': '+ rxn)
        for k in range(n_expert):
            
                  
            config = {
                      'HF_HOME_PATH':'/global/cfs/cdirs/m410/haoteli/LLaMA/',
                      'base_model_path': 'unsloth/Meta-Llama-3.1-8B-Instruct',
                      'finetune_adaptor_path': '/global/cfs/cdirs/m410/haoteli/LLaMA/Expert_Model_Checkpoints_SFP/',
                      'rxn':rxn,
                      'rxn_index':i+1,
                      'expert_id':cell_ids[i][k],
                      'beam_size':beam_size,
                      'beam_group':beam_group,
                      'return_sequence':return_sequence,
                      'diversity_penalty':diversity_penalty,}
            
            assert os.path.exists(os.path.join(config['finetune_adaptor_path'], 'Expert_' + str(config['expert_id']))), print('Assertion Error: Expert ' + str(config['expert_id'])+ ' does not exist!')
            
            pickle.dump(config,open('prediction.config','wb')
                       )
            
            os.system('sh /global/cfs/cdirs/m410/haoteli/LLaMA/Pipeline_Prediction_From_Mixture_Expert/run_prediction.sh > output')
            os.system('rm output')
            
            all_valid_predictions = pickle.load(open('Prediciton_from_{expert}.pkl'.format(expert = cell_ids[i][k]), 'rb'))
            os.system('rm Prediciton_from_{expert}.pkl'.format(expert = cell_ids[i][k]))
            
            
            
            
            for pred_index,pred in enumerate(all_valid_predictions):
                
                smarts = get_smarts_from_pred_text(pred)
                
                try:
                    show_rxn(smarts)
                except:
                    print('extracted invalid SMARTS:{smarts}'.format(smarts = smarts))
                
                
                print('Expert: ' + str(k+1) + '\n' + 'Prediction:'+str(pred_index+1))
                print()
                print('Reagents (SMILES):' + pred.split('Reagents (SMILES):')[-1])
                
                
                if show_reference:
                    print()
                    print()
                    procedure = pred.split('Procedure:\n')[1].split('\n')[0]
                    
                    sub_df = df[df['quadrant'].values == cell_ids[i][k]]
                    reference_distances = get_edit_distance(procedure, sub_df)
                    
                    distance_index = np.argsort(reference_distances)
                    
                    assert len(distance_index) >= n_references
                    
                    for j in range(n_references):
                        print('reference {ref_index} \ndistance: {dist} \nreference reaction: {rxn}'.format(
                            ref_index = j+1,
                            dist = reference_distances[distance_index[j]],
                            rxn = sub_df.iloc[distance_index[j]].reaction,
                            
                        ) 
                             )
                        
                        try:
                            print('Displaying Possible Reference Reaction')
                            show_rxn(sub_df.iloc[distance_index[j]].smiles)
                        except:
                            print('Refernce Reaction Invliad')
                            
                        
                        
                        display_highlighted_diff(sub_df.iloc[distance_index[i]]['paragraphText'], procedure)
                        
                    
                    