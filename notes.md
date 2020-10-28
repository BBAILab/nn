# Revisions to Original Code for Publishing on GitHub

Modifications to the original code were necessary in order to adhere to GitHub file size restrictions and to modify machine-specific file paths so that the code 
is more likely to run upon download.

The training of the Madry model originally included

The genetic algorithm code was modified from its original state to remove machine specific folder navigation options so that it 
would be more generally usable upon download.  HEre are the chagnes that were made.

- ga_control.py
  - Comment out Lines 51-54: these created a specific subfolder for output depending on whther the code was running on the HPC or an office machine
  - Comment out Line 59, add Line 60: remove the use of the subfolder as defined in the lines referenced in the bullet above
  - Comment out Line 20, change path for worker program and instantiate in Line 21