# SUSP-EYE-CIOUS
THREAT DETECTION WITH FACIAL EXPRESSIONS


Dependencies:
1. Tensorflow
2. Numpy, Argparse
3. OpenCV
4. Keras
5. Glob
   Install all modules before running.


FOR CUSTOM TRAINING WEAPON DATASET:

1: (Optional) Add your Images in Tensorflow/workspace/training_demo/images/train folder. 
2: (Optional) Generate .xml files which contain bounding box information using LabelImg tool which is present in root 
   directory.
   // Optional if you want to continue with the same dataset as ours. (More training would lead to more correct results).
   // Our model is trained for 17,000+ iterations. 
3: Move to Directory (tensorflow_env) root\Tensorflow\workspace\training_demo>
4: Run the command
   python train.py --logtostderr --train_dir = training/ --pipeline_config_path=training/ssd_inception_v2_coco.config
   Run till the loss reach less than 1.3 or number of iterations cross 2,00,000.

FOR RUNNING Threat Detection System:

1: Move to Directory root\Tensorflow\workspace\training_demo>
2: Run the command
   python MAIN_WEBCAM_FILE_RUNNING_LATEST_MODIFIED_14_5_19.py (for Webcam)
OR python MAIN_VIDEO_INPUT_FILE_RUNNING_LATEST_MODIFIED_14_5_19.py --input <path_to_input>/<input_file_name> --output <path_to_output>/<output_file_name>
   (for processing input file) (Input can be any video file)
