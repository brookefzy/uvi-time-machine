# Data Organization Folder
This folder contains scripts to organize the data

* **STEP 0.** 00_transfer.ipynb: this script transfer data between different server
* **STEP 1.** 01_downloadpipeline.ipynb (01b): download roadnetwork, gsv file used in this study
* **STEP 2.** 02_summary.ipynb: count gsv, drop duplicates
* **STEP 2b.** 02_clean_up_pano.py: assign ring to each pano_id
* **STEP 3.** 03_filter_data.py: get the GSV size. Size being too small are either corrupted or no content


