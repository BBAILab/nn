# Revisions to Original Code for Publishing on GitHub

The genetic algorithm code wqs modified from its original state to remove machine specific folder navigation options so that it 
would be more generally usable upon download.  HEre are the chagnes that were made.

- ga_control.py
  - Comment out Lines 51-54: these created a specific subfolder for output depending on whther the code was running on the HPC or an office machine
  - Comment out Line 59, add Line 60: remove the use of the subfolder as defined in the lines referenced in the bullet above
  - Comment out Line 20, change path for worker program and instantiate in Line 21