# Notes on data processing for journal article

The files in this folder ("JournalArticle") are the files used to generate the datasets and the machine learning models used in the journal article. Files are copied from other folders and a number is added to the filename to indicate the processing order. The main purpose of this is to have a clear "trail" of code from raw data to finished data presented in the article.

Note that not all processing of data is done through Python code / notebooks. Initial processing of hyperspectral data is done using [Spectronon](https://resonon.com/software). In this case, initial processing involves

- Converting from raw data ("counts") to radiance (data saved in folder "2_Radiance")
- Georeferencing (data saved in folder "3_Rad_Georef"). 

Georeferenced images were plotted in QGIS together with an RGB base map (generated from an RGB dataset using Pix4D), and was analyzed together with sets of underwater images collected along transects. Based on this, the georeferenced RGB versions of the SGC hyperspectral images (see point 1 below) were annotated using [Hasty](hasty.ai). Annotations were exported from Hasty and saved in folder "M_Annotation".

The processng performed by the files in this folder are based on the annotations and the georeferenced hyperspectral radiance images. Numbering in the list below corresponds to numbering of the files.

1. The hyperspectral images are "sun glint corrected" using [Hedley's method](https://doi.org/10.1080/01431160500034086). The results are saved as ENVI files. Note that these are limited to wavelengths below 750 nm. Sun glint corrected files were also rendered as RGB images (with inpainting to fill in missing pixels), to be used for annotation. 
2. All spectra from all annotated areas are collected and saved as Tensorflow datasets (one dataset per hyperspectral image), in folder 4b_Rad_Georef_SCG_Spectra
3. Spectra collected in 2. are used to create / "train" a PCA model. First, 20% of spectra are randomly collected, and the, these spectra are undersampled to balance the number of spectra per class. The PCA model is saved in folder "M_PCA_Model".
4. The PCA model is used to create "PCA images", which have the same pixels as the hyperspectral images, but fewer channels. These images are "inpainted" to fill in pixels with missing data. Results are saved in folder "5a_Rad_Georef_SGC_PCA".
5. PCA images are split into training and validation images (this was done manually, by moving files into separate folders). The PCA images are split into 128x128 pixel "tiles" (only tiles with more than 5% of pixels annotated are included. This reduces the number of unannotated pixels included in the analysis). PCA "spectra" are also extracted from the images and saved as separate datasets.  
6. Train random forest and SVM models on PCA spectra (single pixels from PCA images). Models are saved to folder X_SavedModels_RF_SVM. 
7. Train U-Net (based on Keras / Tensorflow) on PCA tiles. Was tested on several versions with different "depths" (model sizes), and different learning rate adjustments (constant / reduce on plateau). 
8. Compare results for RF, SVM and U-Net models. Results are shown both as whole images (3 validations images) and as confusion matrices. Final models used were 
    - Unet (depth 2, ReduceLROnPlateau): unet_model.epoch40-loss0.028375-acc0.887.hdf5
    - Random forest: 20210825_OlbergAreaS_RF_InpaintedDataset_n-est_20_min-samp-leaf_15_max-samp_0.6.pkl
    - SVM: 20210825_OlbergAreaS_SVM_InpaintedDataset_samp-frac_0.2_C_0.5_kernel_rbf_gamma_scale.pkl'




