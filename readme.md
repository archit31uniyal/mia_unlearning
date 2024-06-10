In order to run the code, we need to follow the following steps:

- Create a conda environment with python 3.10 using the following command: **conda create -n <environment_name> python=3.10.13**
- Activate the environment: **conda activate <environment_name>**
- Install dependencies: **pip install -r requirements.txt**
- Running *attack.py* to generate data
  
  **CUDA_VISIBLE_DEVICES=0,1,2,3 python attacks.py --output_dir <path_to_output_dir> --target_model <target_model_path> --member1 <forgotten_data_path> --key1 <text_column_name> --member2 <member2_data_path> --key2 <text_column_name> --nonmember <nonmember_data_path> --key <text_column_name> --max_length <max_sequence_length> --ref_model <reference_model_path> --seed <integer_value> --recompute**

- Run *analysis.ipynb* with kernel set to *<environment_name>*