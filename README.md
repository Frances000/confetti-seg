Last Updated 15th Aug 2025.
Author: Frances Kan (fkan0396)

This repository stores the code for a Python machine learning segmentation pipeline. The main files cover a workflow for image selection, 
feature generation, and distance map incorporation. 

confetti-seg/02_src/gui_tongue_roi.py

    Launches GUI for users to define the regions of interest. Random subregions 
    are then selected automatically depending on the NUM_PATCHES_PER_REGION 
    parameter, which is by default set to 5. 

    Output structure:
      output/
        filiform_001/
          C/
          Composite/
          G/
          R/
          Y/
        foliate_L_002/
          C/
          Composite/
          G/
          R/
          Y/
        foliate_R_003/
          C/
          Composite/
          G/
          R/
          Y/

confetti-seg/02_src/threshold_stack.py

    Build TWS-like feature stacks for pre-separated colour channels (C, R, G, Y),
    incorporating features similar to those used in Bettega's 2022 Honours Thesis,
    and for every plane in the stack, append two binarised variants:
      - Otsu (global, automatic)
      - Percentile-14% (top 14% brightest pixels = foreground)
    
    Distance maps are NOT generated here.
    
    Usage:
      python make_feature_stacks_split_channels.py \
          --source "/path/to/source_root" \
          --dest   "/path/to/dest_root"
    
    Expected source structure:
      source_root/
        C/  *.tif|*.png|...
        R/  ...
        G/  ...
        Y/  ...
    
    Outputs:
      dest_root/
        C/<basename>_Feature-stack0001.tif (+ *_labels.txt)
        R/...
        G/...
        Y/...
    
    Requirements:
      pip install numpy opencv-python scikit-image tifffile tqdm
