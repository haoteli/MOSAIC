# This training script is executed repeatedly with detection of the previous training progress
# In a Slurm system, it is recommended to create dependencies of jobs so that they execute sequentially.

import os
# Set environment variables and seeds
os.environ["HF_HOME"] = "path/to/your/intented/directory/"

import torch
torch.manual_seed(225)


from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


model_name = 'llama-3.1-8B-Instruct' 
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

#Set up LoRA configuration
rank = 16
peft_config = LoraConfig(
    r=rank,
    
    lora_alpha=rank*4,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # these moduals must be spcified explicitly
    lora_dropout=0.00, 
    bias="none",
    task_type="CAUSAL_LM",
)

#Apply LoRA
model = get_peft_model(model, peft_config)


import pickle
import pandas as pd
df = pickle.load(open('FPCompatible_Cleaned_Pistachio.pkl','rb'))

df = df[~df['paragraphText'].isna()].reset_index(drop=True) # If there are no descriptions, drop them
df = df[~(df['agent'] == '[]')].reset_index(drop=True) 
df = df[~(df['agent_name'] == '[]')].reset_index(drop=True)
df = df[~(df['solvent'] == '[]')].reset_index(drop=True)
df = df[~(df['solvent_name'] == '[]')].reset_index(drop=True) 
df = df[~(df['yield'] == '[]')].reset_index(drop=True)
df = df[['Example' not in p for p in df['paragraphText']] ].reset_index(drop=True)

df = df.sample(frac=1).reset_index(drop=True) # For the same dataset, shuffle it everytime the training is executed.

print('Shuffled Dataset')


df = df[['product','reactant','smiles' , 'agent', 'solvent',
         'product_name', 
         'reactant_name',
         'agent_name',
         'solvent_name',
         'paragraphText', 'namerxndef', 'yield']]


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

datasize = len(df)//5 # Deviding the datasets into 10 partitions, each partition takes around 12 hrs, submit 20 jobs = 1 epoch.

print('Sampling Dataset Size:',datasize)

dataset_before_transformation = Dataset.from_pandas(df.iloc[0:datasize])

dataset = dataset_before_transformation.map(formatting_prompts_func, batched=True)

print(dataset['text'][0])

batch_size = 4

output_folder_directory = "haoteli/LLaMA/" + model_name
# If you dont have the directory, it will be automatically created for you. So no worries.

# Set up training arguments
training_args = TrainingArguments(
    output_dir=output_folder_directory,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    num_train_epochs=50, 
    learning_rate=2e-4,
    fp16=True,
    optim="adamw_torch_fused",
    #max_grad_norm=0.,
    warmup_ratio=0.01, # 50*0.01 = 0.5 epoch = (1M/4)steps per epoch * 0.5 epoch = 125,000 steps.
    group_by_length=True,
    lr_scheduler_type="cosine",
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
print('Continuing training from {}'.format(latest))
trainer.train(
  
      resume_from_checkpoint=latest
  
 )

