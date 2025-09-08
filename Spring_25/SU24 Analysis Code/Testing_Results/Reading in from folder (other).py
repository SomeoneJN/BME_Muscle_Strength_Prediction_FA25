# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:30:08 2025

@author: mgordon
"""

import os
import re
import csv

def extract_numbers_to_dict(filename):
  """
  Extracts numbers enclosed within parentheses from a filename 
  and stores them in a dictionary with keys like 'Part1_E', 'Part3_E', etc.

  Args:
    filename: The filename to process.

  Returns:
    A dictionary where keys are the prefixes (e.g., 'Part1_E') 
    and values are the corresponding extracted numbers.
  """
  pattern = r'(\w+_E)\(([0-9.]+)\)'
  matches = re.findall(pattern, filename)
  return {key: float(value) for key, value in matches}

def get_filenames(folder_path):
  """
  Gets a list of filenames within a specified folder, filtering for .feb files.

  Args:
    folder_path: The path to the folder.

  Returns:
    A list of .feb filenames within the folder.
  """
  try:
    filenames = os.listdir(folder_path)
    return [f for f in filenames if f.endswith(".feb")] 
  except FileNotFoundError:
    print(f"Error: Folder not found at {folder_path}")
    return []

def process_folder(folder_path):
  """
  Processes files in a folder, extracts numbers from their filenames, 
  and stores the results in a dictionary.

  Args:
    folder_path: The path to the folder.

  Returns:
    A dictionary where keys are filenames and values are dictionaries 
    containing extracted numbers with their corresponding prefixes.
  """
  filenames = get_filenames(folder_path)
  results = {}
  for filename in filenames:
    full_filepath = os.path.join(folder_path, filename)
    results[filename] = extract_numbers_to_dict(filename)
  return results

# Example usage
folder_path = r"C:\Users\mgordon\Downloads\Runs for Testing-20250108T014256Z-001\Runs for Testing" 
extracted_data = process_folder(folder_path)


dictionary_file = r"C:\Users\mgordon\Downloads\Runs for Testing-20250108T014256Z-001\Runs for Testing\feb_variables.csv" #DONE

run_file = open(dictionary_file)
DOE_dict = csv.DictReader(run_file)


# Print the extracted data
for filename, numbers in extracted_data.items():
  print(f"Filename: {filename}")
  for prefix, value in numbers.items():
    print(f"  {prefix}: {value}")
  print() 