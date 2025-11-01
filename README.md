Last Updated 15th Aug 2025.
Author: Frances Kan (fkan0396)

This repository stores the code for a Python machine learning segmentation pipeline. The main files cover a workflow for image selection, 
feature generation, and distance map incorporation. 

# Pipeline Workflow
## Stage 1: Pseudoimage Generation
1. Starting with an initial brightfeld image, regions of interest are selected for generation of pseudoimages. Run "python3 gui_tongue_roi.py", and exit once the selection is complete. 

python3 gui_tongue_roi.py

This launches GUI for users to define the regions of interest. Random subregions 
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

2. Create a feature stack using "confetti-seg/02_src/threshold_stack.py". This generates both the distance stacks and the base stack for each input image, as well as a .txt file with the labels of each layer in the stack. As multiple distance-based features are created for each initial filtered image, distance stacks take ~3x longer to generate. 

python3 threshold_stack.py --source "/path/to/source_root" --dest "/path/to/dest_root"

Build TWS-like feature stacks for pre-separated colour channels (C, R, G, Y),
incorporating features similar to those used in Bettega's 2022 Honours Thesis,
and for every plane in the stack, append two binarised variants:
  - Otsu (global, automatic)
  - Percentile-14% (top 14% brightest pixels = foreground)

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

3. Images are converted to a csv format for input into training. Run create_csv_stack.py, specifying the root directory containing the channel subfolders with Base_*.tif and *_labels.txt files as a command line parameter, as well as the output directory where the csvs are intended to be generated. 

python3 create_csv_stack.py --root "path/to/stack_root" --dest "path/to/csv_output_dest" 

Compose per-patch flattened CSVs from existing Base feature stacks.  
Columns:   
   [ class,     
        <all non-mask slices from C (with _cyan suffix)>,     
        <all non-mask slices from G (with _green suffix)>,     
        <all non-mask slices from R (with _red suffix)>,     
        <all non-mask slices from Y (with _yellow suffix)> ]  

'class' construction:   
   per pixel, read each channel's mask plane (first slice, or label 'mask');   
   value 0 -> add nothing; 127 -> add one letter; 255/254 -> add two letters.   
   After C,G,R,Y processed: if len<2, pad with 'B'; then alphabetise (e.g. 'YR'->'RY').


## Stage 2: Training on Pseudoimages
1. We train two models for the segmentation of Confetti channels. For training, we use two files. The first model uses distance-mapped CSVs - 02_src/no_selection_distance_training.py, and the second uses the base CSVs - 02_src/no_selection_base_training.py. We recursively search input folders for the csvs and output the trained models, as well as the features used in training. 


python3 /scratch/wb85/fk3921/final_stack_csv/mouse_tongue_subimages/no_selection_distance_training.py \
  --roots \
  "/scratch/wb85/fk3921/final_stack_csv/confetti-seg-training-reduced/final_csv/mouse_tongue_subimages/Cd04M-155706-20170601-merge-BF" \
  "/scratch/wb85/fk3921/final_stack_csv/confetti-seg-training-reduced/final_csv/mouse_tongue_subimages/Cd04M-155706-20171025-merge-BF" \
  "/scratch/wb85/fk3921/final_stack_csv/confetti-seg-training-reduced/final_csv/mouse_tongue_subimages/Cn03F-168849-20170714-merge-BF" \
  "/scratch/wb85/fk3921/final_stack_csv/confetti-seg-training-reduced/final_csv/mouse_tongue_subimages/Cn12F-168805-20170814-merge-BF" \
  --outdir "/scratch/wb85/fk3921/final_stack_csv/reduced_stack_no_selection_distance_trained_models"

python3 /scratch/wb85/fk3921/final_stack_csv/mouse_tongue_subimages/no_selection_distance_training.py \
  --roots \
  "/scratch/wb85/fk3921/final_stack_csv/mouse_tongue_subimages/Cd04M-155706-20170601-merge-BF" \
  "/scratch/wb85/fk3921/final_stack_csv/mouse_tongue_subimages/Cd04M-155706-20171025-merge-BF" \
  "/scratch/wb85/fk3921/final_stack_csv/mouse_tongue_subimages/Cn03F-168849-20170714-merge-BF" \
  "/scratch/wb85/fk3921/final_stack_csv/mouse_tongue_subimages/Cn12F-168805-20170814-merge-BF" \
  --outdir "/scratch/wb85/fk3921/final_stack_csv/no_selection_distance_trained_models"

Train two models for Confetti segmentation features: 
   1) distance-mapped CSVs (filenames starting with 'distance_') 
   2) base CSVs (same folders, filenames without the 'distance_' prefix) 

- Recurses date folders like: 
   /.../Cd04M-155706-20170601-merge-BF/ 
       ├─ filiform_001_csv/     
              ├─ distance_filiform_patch_00.csv 
              └─ filiform_patch_00.csv 
       ├─ foliate_L_002_csv/ 
       └─ foliate_R_003_csv/ 

Outputs: 
   outdir/ 
       model_base.joblib 
       selector_base.joblib 
       model_distance.joblib 
       feature_names_*.txt

## Stage 3: Testing on Pseudoimages
1. Testing on pseudoimages is carried out using the 02_src/convert_test_to_image.py file, which creates visualisations of the 
diff between the prediction and ground truth for the base and distance variant of each model. It also creates an output summary with statistical measures of accuracy, a legend for reconstruction of images. To write image tiffs, you must provide the shape (dim) of the input image. The parameters required are the model path and the csv path with the ground truth, as well as features. A basename for the prediction tiffs can be supplied as --pred_tiff, and --out can be used to overwrite the default txt report path. 

For example,
python3 02_src/convert_test_to_image.py --model_path "/Volumes/Lyons_X5/distance_inclusion_variation/confetti-seg-training-reduced/all_dt_models/reduced_stack_no_selection_distance_trained_models/model_distance_rf.joblib" --csv_path /Volumes/Lyons_X5/distance_inclusion_variation/confetti-seg-training-reduced/final_csv/mouse_tongue_subimages/Fa01M-156853-20170613-merge-BF/foliate_L_002_csv/distance_foliate_L_patch_00.csv --shape 200x100

2. Summarising statistics for pseudoimage testing can be achieved with 02_src/summarise_eval_txt.py, which outputs an aggregated eval report (csv) reporting accuracy, miscalss rate, kappa stat, macro precision, weighted precision, f1, macro f1, eval runtime, features, pixels, rows.

python3 02_src/summarise_eval_txt.py --inputs [directory to be searched recursively/file] --outdir [where to write the csv]

## Stage 4: Testing on Real Images - No Ground Truth
Note: for practical usage, modify the parameters in 02_src/run_confetti_timeseries.sh. This enables longitudinal tracking later on, or optionally the conversion of a time series into predictions/segmentations. However, this script operates on the following (Explained through single images rather than series)
1. From an input tiff (composed of multiple channels), we can specify a central region to crop. We then separate by channel (Red, cyan, green, yellow), and save these in respective folders. By default, we assume the third channel is the BF image, but this can be overwritten. 

python 02_src/split_confetti_channels.py \
  --image "/Users/franceskan/Downloads/Fa01M-156853-20170613-merge.tif" \
  --dest_root "/Volumes/Lyons_X5/real_confetti_test/Fa01M-156853-20170613" \
  --crop 800x700 \
  --include_bf \
  --chnum_map "1=R,2=G,3=BF,4=C,5=Y" \
  --report_json "/Volumes/Lyons_X5/real_confetti_test/Fa01M-156853-20170613/reports/split_report.json" \
  --overwrite

2. Stacks are generated from these input images: 

python3 /Users/franceskan/Documents/confetti-seg/02_src/threshold_stack.py \
--source /Volumes/Lyons_X5/real_confetti_test/Fa01M-156853-20170613 \
--dest /Volumes/Lyons_X5/real_confetti_test/stack_Fa01M-156853-20170613

3. From image stacks, generate csv (same method as used in training phase)

python 02_src/create_csv_stack.py \
  --root /Volumes/Lyons_X5/real_confetti_test/stack_Fa01M-156853-20170613 \
  --out  /Volumes/Lyons_X5/real_confetti_test/csv_Fa01M-156853-20170613 \
  --mode infer

4. Run ML prediction on the input csv stack and create the image visualisation.
python3 02_src/convert_test_to_image.py \
--model_path "/Volumes/Lyons_X5/confetti-seg/gadi_models/no_selection_distance_trained_models/model_distance_rf.joblib" \
--csv_path /Volumes/Lyons_X5/real_confetti_test/csv_Fa01M-156853-20170613/distance_Fa01M-156853-20170613-merge.csv \
--shape 800x700

5. Segment regions
python3 /Users/franceskan/Documents/confetti-seg/02_src/segment_clones.py \
  --label_tiff /Volumes/Lyons_X5/confetti-seg/real_image_csv_output/distance_filiform_patch_00__distance_rf__preds_labels.tif \
  --legend_csv /Volumes/Lyons_X5/confetti-seg/real_image_csv_output/distance_filiform_patch_00__distance_rf__preds_legend.csv \
  --csv_out    /Volumes/Lyons_X5/confetti-seg/real_image_csv_output/Fa01M_patch_00__distance_rf__clone_metrics.csv \
  --overlay_out /Volumes/Lyons_X5/confetti-seg/real_image_csv_output/Fa01M_patch_00__distance_rf__clone_overlay.tif \
  --min_pixels 10 \
  --pix_size_um 0.5 \
  --core_radius 2

## Stage 5: Confetti Back-Tracer GUI (Using Tkinter, Random Forest model on distance-variant features)
Loads images named: Aa01F-145201-YYYYMMDD-merge.tif (or .tiff), extracts date from the filename and sorts ascending. Tri-panel viewer shows (t, t-1, t-2) with latest on the right. First ROI locks (W,H); later frames auto-propagate both ROIs and allow repositioning. Two regions per timepoint: Tumour (red) and Normal (green). Builds distance-variant feature planes in-memory (percentile & Otsu binaries, skeletons, distance maps) using the same logic as the user's stack generator. RandomForest (.joblib) inference over both ROIs; computes simple metrics. No CSV written unless --export_csv is provided; JSON session is always saved

python 02_src/rf_backtrace_gui_distance.py \
    --images_dir /Volumes/Lyons_X5/real_confetti_test/4NQO_stitched/Aa01F-145201-stitched_gui_test_copy \
    --rf_model /Volumes/Lyons_X5/distance_inclusion_variation/confetti-seg-training-reduced/all_dt_models/reduced_stack_no_selection_distance_trained_models/model_distance_rf.joblib
