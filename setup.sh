#!/bin/bash  
echo "Creating virtual environment..."  
# Exit immediately if a command exits with a non-zero status.  
set -e  

# Create a virtual environment named "Deep Learning For Image Analyses"  
echo "Creating virtual environment 'Deep_Learning For_Image_Analyses'..."  
python3 -m venv "Deep_Learning_For_Image_Analyses"  

# Activate the virtual environment  
echo "Activating virtual environment..."  
source "Deep_Learning_For_Image_Analyses/bin/activate"  

# Install requirements  
echo "Installing requirements from requirements.txt..."  
pip install --upgrade pip  
pip install -r requirements.txt  

echo "Virtual environment 'Deep Learning For Image Analyses' created and requirements installed!"  
echo "Remember to activate the environment before running your scripts: source 'Deep Learning For Image Analyses/bin/activate'"  