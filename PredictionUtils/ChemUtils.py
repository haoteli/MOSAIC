from rdkit import Chem
from rdkit.Chem import RDKFingerprint, AllChem
from rdkit.Chem import AllChem,DataStructs
from rdkit.Chem import Draw
from IPython.display import display, Image
import numpy as np

def create_rxn_Mix_FP(rxn, rxnfpsize=1024, pfpsize=1024, useFeatures=False, calculate_rfp=True, useChirality=False):
    
    fpgen = AllChem.GetRDKitFPGenerator(maxPath=4,fpSize=rxnfpsize)
    
    rsmi = rxn.split('>>')[0]
    psmi = rxn.split('>>')[1]
    rct_mol = Chem.MolFromSmiles(rsmi)
    prd_mol = Chem.MolFromSmiles(psmi)
    
    rsmi = rsmi.encode('utf-8')
    psmi = psmi.encode('utf-8')
    try:
        mol = Chem.MolFromSmiles(rsmi)
    except Exception as e:
        print(e)
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(
            mol=mol, radius=2, nBits=rxnfpsize, useFeatures=useFeatures, useChirality=useChirality)
        fp = np.empty(rxnfpsize, dtype=int)
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
        
        rdkitfp = np.array(fpgen.GetFingerprint(mol).ToList())
        fp = np.concatenate((rdkitfp,fp), axis = -1) # RDKIT(p=4)+ Morgan(r=2) fingerprint
        
    except Exception as e:
        print("Cannot build reactant fp due to {}".format(e))
        return
    rfp = fp
    
    try:
        mol = Chem.MolFromSmiles(psmi)
    except Exception as e:
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(
            mol=mol, radius=2, nBits=pfpsize, useFeatures=useFeatures, useChirality=useChirality)
        fp = np.empty(pfpsize, dtype=int)
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
        rdkitfp = np.array(fpgen.GetFingerprint(mol).ToList())
        fp = np.concatenate((rdkitfp,fp), axis = -1)
        
    except Exception as e:
        print("Cannot build product fp due to {}".format(e))
        return
      
    pfp = fp
    return [pfp, rfp, pfp-rfp]      

  

def canonicalize(smiles, keep_stereo_info = True):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles = keep_stereo_info)
    
def convert_smiles_to_prompt_format(rxn, keep_stereo_info = True):
    # reactant and product are "." separated
    
    reactants,product = rxn.split('>>') # Spliting arbitrary reaction pasted from ChemDraw
    
    
    reactants = reactants.split('.') # Further splitting by '.' 
    products = product.split('.') # Even if the product side does not have multiple molecules, this will simply return a list with a single product
    
    reactants = [canonicalize(r,
                              keep_stereo_info) for r in reactants]
    
    products = [canonicalize(p,
                             keep_stereo_info) for p in products]
    
    # return the list of canocalized reactants and products that can be used for prompt
    return(reactants,products)
  
  

def pass_check(answer):
    # Making sure that each required field is in the output, essentially no hallucination/repetitions should be accounted
    try:
        #assert "Reaction SMARTS" in answer
        assert "Reagents (SMILES)" in answer
        assert "Solvent (SMILES)" in answer
        assert "Product Name" in answer
        assert "Reactant Name" in answer
        assert "Solvent Name" in answer
        assert "Reaction Procedure" in answer
        assert "Reaction Name and Classification" in answer
        assert "Yield and Characterization" in answer
        
        return True
    
    except:
        return(False)
      
def show_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    drawer = Draw.rdDepictor.SetPreferCoordGen(True)  # Use CoordGen for 2D depiction
    d2d = Draw.MolDraw2DCairo(400, 400)  # Specify canvas size
    d2d.drawOptions().addAtomIndices = True  # Show atom indices
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    display(Image(d2d.GetDrawingText()))
    print("Display molecule")
def show_rxn(rxn_str):
  # rxn_str should be "SMILES or SMARTS>Reagents>SMILES or SMARTS"
    rxn = AllChem.ReactionFromSmarts(rxn_str, useSmiles = True)
    d2d = Draw.MolDraw2DCairo(2500,500)
    d2d.DrawReaction(rxn, highlightByReactant=True)
    #d2d.DrawReaction(rxn)
    display(Image(d2d.GetDrawingText()))

      