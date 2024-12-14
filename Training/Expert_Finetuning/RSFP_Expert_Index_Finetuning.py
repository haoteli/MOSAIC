# this script takes in an argument "expert_index" an integer and perform the training.
import os
# Set environment variables and seeds
os.environ["HF_HOME"] = "path/to/your/intented/directory/"

import torch
torch.manual_seed(225)


from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model, PeftConfig
from trl import SFTTrainer
import sys

expert_index = int(sys.argv[-1])

# Load tokenizer and model
base_model_name = 'llama-3.1-8B-Instruct' 
model_name = 'llama-3.1-8B-Instruct/checkpoint-750000' # This is the checkpoint from the first finetuning
tokenizer = AutoTokenizer.from_pretrained(model_name)



base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    #low_cpu_mem_usage=True,
    local_files_only = True,
)

# Load the previous LoRA configuration
peft_config = PeftConfig.from_pretrained(model_name)

# Load the previous LoRA weights
# is_trainable is the key to making the model continually trainable!
model = PeftModel.from_pretrained(base_model, model_name,is_trainable = True)




import pickle
import pandas as pd
df = pickle.load(open('RSFP_Train_Quadrant_df.pkl','rb'))


# Define the prompt template
single_step_reaction_prompt = """### Input:
Product (SMILES): {product}
Reactant (SMILES): {reactant}
Reaction SMARTS: {smiles}
Reagents (SMILES): {agent}
Solvent (SMILES): {solvent}
Product Name: {product_name}
Reactant Name: {reactant_name}
Reagent Name: {agent_name}
Solvent Name: {solvent_name}

### Response:
Reaction Procedure:
{paragraphText}

Reaction Name and Classification:
{namerxndef}

Yield and Characterization:
{yields}
"""




EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN



def formatting_prompts_func(examples):
    _product = examples["product"]
    _reactant = examples["reactant"]
    _smiles = examples["smiles"]
    _agent = examples["agent"]
    _solvent = examples["solvent"]
    
    _product_name = examples["product_name"]
    _reactant_name = examples["reactant_name"]
    _agent_name = examples["agent_name"]
    _solvent_name = examples["solvent_name"]
    
    
    _paragraphText = examples["paragraphText"]
    _namerxndef = examples['namerxndef']
    _yield = examples['yield']
    
    
    
    texts = []

    for (
        product,
         reactant, 
         smiles, 
         agent,
         solvent,
         product_name,
         reactant_name,
         agent_name,
         solvent_name,
         paragraphText,
         namerxndef,
         yields
        ) in zip(_product,
                                                           _reactant,
                                                           _smiles, 
                                                           _agent,
                                                           _solvent,
                        
                                                           _product_name,
                                                           _reactant_name,
                                                           _agent_name,
                                                           _solvent_name,
                        
                                                           _paragraphText,  
                                                           _namerxndef,
                                                           _yield
                                                          ):
        # Format the text using the retrosynthesis prompt template
        text = single_step_reaction_prompt.format(
            product=list(set(eval(product))),
            reactant = list(set(eval(reactant))),
            smiles = smiles,
            agent = list(set(eval(agent))),
            solvent = list(set(eval(solvent))),
            product_name = list(set(eval(product_name))),
            reactant_name = list(set(eval(reactant_name))),
            agent_name = list(set(eval(agent_name))),
            solvent_name = list(set(eval(solvent_name))),
            paragraphText=paragraphText,
            namerxndef=namerxndef,
            yields=yields
        )
        # Add EOS_TOKEN if needed (you'll need to define this based on your tokenizer)
        text += EOS_TOKEN
        texts.append(text)

    return {"text": texts}


from datasets import Dataset

df = df[df['quadrant'] == expert_index] # Subindexing to the expert-specific domain

dataset_before_transformation = Dataset.from_pandas(df)

dataset = dataset_before_transformation.map(formatting_prompts_func, batched=True)

print('='*50)

print('This is expert {exp}, expert {exp} takes care of the following domain knowledge distribution'.format(exp = expert_index))

displaydict = dict(df['namerxndef'].value_counts())
for item in displaydict.items():
    print(item)
    
print('='*50)
print(dataset['text'][0])

batch_size = 4

output_folder_directory = "Expert_Model_Checkpoints_RSFP/Expert_" + str(expert_index)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=output_folder_directory,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    num_train_epochs=10, 
    learning_rate=2e-4,
    fp16=True,
    optim="adamw_torch_fused",
    warmup_ratio=0.02,
    group_by_length=True,
    lr_scheduler_type= "cosine", 
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=2500,
   
    logging_steps=10,
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=2048,
)




def get_latest_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    return os.path.join(output_dir, max(checkpoints, key=lambda x: int(x.split("-")[1])))
  
 

latest = get_latest_checkpoint(output_folder_directory)
if latest is not None:
  
  
  print('='*50)
  print('Already Trained')
  os._exit(0) # exit
 

else:
  print('='*50)
  print('Did not find previous checkpoint, training new model')
  trainer.train() # If this is the first time, else, resume it from the checkpoint

os._exit(0) # terminate

         