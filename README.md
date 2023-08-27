# neurmorph-long-pulses
Code files for the article titled "Towards neuromorphic computing using longitudinal pulses in real fluids"

This reposity includes the following python files and the yeast dataset:

1. The file "yeast.data" includes 1484 instances with 8 attributes. It was downloaded from https://archive.ics.uci.edu/dataset/110/yeast under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given. Reference is "A Knowledge Base for Predicting Protein Localization Sites in Eukaryotic Cells", Kenta Nakai & Minoru Kanehisa, Genomics 14:897-911, 1992.
2. The file "Class_mainYeast.py" is the main script to perform classification of the yeast data. The script loads the yeast data file, runs it through the reservoir and generates the H matrix.
3. The file "Class_readData.py" reads the Yeast data from the file "yeast.data"
4. The file "Class_reservoir" uses the data from "yeast.data" to stimulate the reservoir (vdW fluid model) and saves the solution of the fluid fields as an npz data file
5. The file "Class_extractData" extract the output data from the reservoir and builds the H matrix
6. The file "Reg_reservoir.py" applies input into the reservoir and saves the solution of the fluid fields as an npz data file
7. The file "Reg_buildHmatrix.py" uses the reservoir data to build the H matrix
8. The file "Reg_checkData.py" loads the H matrix, removes one data point from the H matrix and tests the quality of prediction using the H matrix on the data point
