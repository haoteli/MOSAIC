from difflib import SequenceMatcher
import re
from IPython.display import HTML, display
import Levenshtein
import numpy as np
import re

def levenshtein_distance(text1, text2):
    # Calculates the levenshtein distance between two texts
    return Levenshtein.distance(text1, text2)

def get_edit_distance(text, df):
  #given a text, compare it to the 'paragraphText field' of the dataframe. Then return the distances.
  distances = []

  for i in range(len(df)):
      p = df.iloc[i]['paragraphText']
      distance = levenshtein_distance(text,p)
      distances.append(distance)
  return(np.array(distances))

def find_partial_text_before_yield(paragraphText):
  all_matches = re.findall(r'\((.*?)\)', paragraphText)
  final_match = None
  valid_matches = []
  for match in all_matches:
    if '%' in match:
        if ',' in match[:match.find('%')]:
          # if comma comes before the percent sign
            final_match = '(' + match + ')'
            
            valid_matches.append(paragraphText[:paragraphText.find(final_match) - 1])

  return(valid_matches)
  
  

def highlight_differences(text1, text2):
    """
    Analyzes and highlights similarities and differences between two text sequences.
    Returns HTML-formatted strings with color-coded highlights.
    
    Green: Matching sequences
    Yellow: Substitutions/replacements
    Red: Unique to text1
    Blue: Unique to text2
    """
    def tokenize(text):
        return re.findall(r'\S+|\s+', text)
    
    def get_opcodes(tokens1, tokens2):
        matcher = SequenceMatcher(None, tokens1, tokens2)
        return matcher.get_opcodes()
    
    def format_html(tokens, tag_start, tag_end):
        return ''.join(tokens[tag_start:tag_end])
    
    # Tokenize input texts
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    # Get comparison opcodes
    opcodes = get_opcodes(tokens1, tokens2)
    
    # Format the output strings with HTML spans
    formatted1 = []
    formatted2 = []
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Matching sequences in green
            span = f'<span style="background-color: #90EE90">{format_html(tokens1, i1, i2)}</span>'
            formatted1.append(span)
            formatted2.append(span)
        elif tag == 'replace':
            # Substitutions in yellow
            formatted1.append(f'<span style="background-color: #FFD700">{format_html(tokens1, i1, i2)}</span>')
            formatted2.append(f'<span style="background-color: #FFD700">{format_html(tokens2, j1, j2)}</span>')
        elif tag == 'delete':
            # Text unique to sequence 1 in red
            formatted1.append(f'<span style="background-color: #FFB6C6">{format_html(tokens1, i1, i2)}</span>')
        elif tag == 'insert':
            # Text unique to sequence 2 in blue
            formatted2.append(f'<span style="background-color: #ADD8E6">{format_html(tokens2, j1, j2)}</span>')
    
    return ''.join(formatted1), ''.join(formatted2)

def display_highlighted_diff(text1, text2):
    """
    Displays the highlighted differences in a Jupyter notebook with a legend.
    """
    highlighted1, highlighted2 = highlight_differences(text1, text2)
    
    html_content = f"""
    <div style="font-family: monospace;">
        <div style="margin-bottom: 10px;">
            <strong>Legend:</strong><br>
            <span style="background-color: #90EE90">----</span> Matching sequences<br>
            <span style="background-color: #FFD700">----</span> Substitutions/replacements<br>
            <span style="background-color: #FFB6C6">----</span> Unique to Reference<br>
            <span style="background-color: #ADD8E6">----</span> Unique to Prediction
        </div>
        <div style="margin-bottom: 10px;">
            <strong>Reference:</strong><br>
            <div style="padding: 5px;">{highlighted1}</div>
        </div>
        <div>
            <strong>Prediction:</strong><br>
            <div style="padding: 5px;">{highlighted2}</div>
        </div>
    </div>
    """
    
    display(HTML(html_content))

    
def get_smarts_from_pred_text(pred_text):
  
  smarts = pred_text.split('Reaction SMARTS: ')[-1] # after the keyword "Reaction SMARTS:" 
  smarts = smarts.split('Reagents (SMILES):')[0] # before the keyword "Reagents (SMILES):"
  if smarts[-1] == '|': #if there is this symbol
    smarts = smarts[:-1]
    smarts = smarts[:smarts.find('|')] # Find the other one and remove the leftoverportion
    
  if smarts[-1] == ' ': # If there is extra space, remove it
    smarts = smarts[:-1]
    
  return smarts