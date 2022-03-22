Assessing radiomics feature stability with simulated CT acquisitions
 
 https://doi.org/10.1038/s41598-022-08301-1


 ------------------------
 
Medical imaging quantitative features were once considered untrustworthy in clinical studies. Nowadays, the advancementof analysis techniques such as machine learning, has enabled these features to be progressively useful in diagnosis andresearch. Tissue characterisation is improved and can be automated via the extraction of quantitative ?radiomic? features fromclinical scans.  Nevertheless, it remains an open question of how to efficiently select such features as they can be highlysensitive to variations in the acquisition details.  In this work, we develop and validate a Computed Tomography simulatorenvironment based on the publicly available www.astra-toolbox.com to facilitate the selection of such features, thus assisting orby-passing costly clinical trials. We show that the variability, stability and descriminative power of the ?radiomics? extracted from the simulator are similar to those observed in a tandem clinical study.

------------------------

 Installation: conda environment.yml 

------------------------

------------------------

# Usage:

run ctsim_main.py with python on gpu,

adjust lists that are passed to simulation function accordigly 
takes a nifti image as input (or dicom) and outputs variable nifty images (=number of total variations)

------------------------

kyriakos.flouris@cantab.net
kflouris@ethz.ch
