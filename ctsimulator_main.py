from __future__ import division

import numpy as np
from os import mkdir
import itertools
from  scipy import ndimage
import astra
import nibabel as nib
from os.path import join, isdir
from imageio import  get_writer


      
def simulation(METHOD, NOISE_LEVEL, ITTERATIONS, NUM_PROJECTIONS, SMOOTHING_SIG, RANDOM_SEED,input_dir,output_dir, phantom_loc, save_projections=False):    
    angles = np.linspace(0, 2 * np.pi, num=NUM_PROJECTIONS, endpoint=False)

    phantom_nifty=nib.load(phantom_loc)
    phantom_arr=np.array(phantom_nifty.dataobj)
    phantom_rows=phantom_arr.shape[0]
    phantom_cols=phantom_arr.shape[1]
    phantom_slices = phantom_arr.shape[2]  # depth.
    print('phantom size:', phantom_arr.shape)
    
    #  detector:
    distance_source_origin = 1000  # [mm]
    distance_origin_detector = 0  # [mm]  
    detector_pixel_size = 1 # [mm]
    detector_rows = 512  # Vertical size of detector [pixels]
    detector_cols = 512  # Horizontal size of detector [pixels]
    phantom= phantom_arr
    
    
    # vol_geom in create expects: (GRIDslice,GRIDrow,GRIDcol)
    if METHOD in ['SIRT', 'FP3D']:
        vol_geom = astra.creators.create_vol_geom(phantom_cols, phantom_slices,
                                    phantom_rows, -phantom_cols/2 ,phantom_cols/2 , -phantom_slices/2,
                                    phantom_slices/2, -phantom_rows/2, phantom_rows/2) 
    elif  METHOD in ['FBP']:
        vol_geom = astra.creators.create_vol_geom(phantom_cols, phantom_slices,
                                            phantom_rows )  

    phantom_id = astra.data3d.create('-vol', vol_geom, data=phantom)
    

    
    
    # Vector cone source
    if METHOD in ['SIRT']:
        pitch=phantom_cols-1 
        vectors = np.zeros((len(angles), 12))
        shift= np.linspace(0, pitch, num=NUM_PROJECTIONS, endpoint=True)
        for i in range(len(angles)):
         	# source
         	vectors[i,0] = np.sin(angles[i]) * distance_source_origin * detector_pixel_size
         	vectors[i,1] = -np.cos(angles[i]) * distance_source_origin * detector_pixel_size
         	vectors[i,2] = 0- pitch/2+shift[i]
        
            # center of detector
         	vectors[i,3] = -np.sin(angles[i]) * distance_origin_detector  * detector_pixel_size
         	vectors[i,4] = np.cos(angles[i]) * distance_origin_detector * detector_pixel_size
         	vectors[i,5] = 0-pitch/2+shift[i]
    
         	# vector from detector pixel (0,0) to (0,1)
         	vectors[i,6] = np.cos(angles[i]) * detector_pixel_size
         	vectors[i,7] = np.sin(angles[i]) * detector_pixel_size
         	vectors[i,8] = 0
        
         	# vector from detector pixel (0,0) to (1,0)
         	vectors[i,9] = 0
         	vectors[i,10] = 0
         	vectors[i,11] = detector_pixel_size		
        
        proj_geom = astra.create_proj_geom('cone_vec', detector_rows, detector_cols, vectors)     


    # Conical source
    elif METHOD in ['FBP', 'FP3D']:
        proj_geom = astra.create_proj_geom('cone', detector_pixel_size, detector_pixel_size, detector_rows, detector_cols, angles,
                              distance_source_origin, distance_origin_detector)

    
    projections_id, projections = astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)

    # Apply Poisson noise. 
    projections=astra.functions.add_noise_to_sino(projections,int(1/NOISE_LEVEL), seed=RANDOM_SEED)

    # Apply smoothening
    for i in range(NUM_PROJECTIONS):
        projection = projections[:, i, :]
        projection = ndimage.gaussian_filter(projection, SMOOTHING_SIG)
        projections[:, i, :]=projection

    
    if save_projections:
        projections_=projections

        if not isdir(input_dir):
            mkdir(input_dir)
        projections_ = np.round(projections_ * 65535).astype(np.uint16)
        
        for i in range(NUM_PROJECTIONS):
            projection = projections[:, i, :]
        
            with get_writer(join(input_dir, 'proj%04d.png' %i)) as writer:
                writer.append_data(projection, {'compress': 9})

    
    print('>>> created projections')
      
    # Copy projection images into ASTRA Toolbox.
    projections_id = astra.data3d.create('-sino', proj_geom, projections)
    
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
   
    if METHOD=='SIRT':

          alg_cfg = astra.astra_dict('SIRT3D_CUDA') 
          alg_cfg['ProjectionDataId'] = projections_id
          alg_cfg['ReconstructionDataId'] = reconstruction_id
          algorithm_id = astra.algorithm.create(alg_cfg)
          astra.algorithm.run(algorithm_id,ITTERATIONS)
        
    elif METHOD=='FBP':
    
        alg_cfg = astra.astra_dict('FDK_CUDA') 
        alg_cfg['ProjectionDataId'] = projections_id
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)
        astra.algorithm.run(algorithm_id,ITTERATIONS)      

    elif METHOD=='FP3D':
        alg_cfg = astra.astra_dict('FP3D_CUDA') 
        alg_cfg['ProjectionDataId'] = projections_id
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)
        astra.algorithm.run(algorithm_id)            
        
       

    
    reconstruction = astra.data3d.get(reconstruction_id)

    
    # Save reconstruction.
    if not isdir(output_dir):
        mkdir(output_dir)
    func = nib.load(phantom_loc)
    ni_img = nib.Nifti1Image(reconstruction, func.affine)
    output_nifty='/{}_nl{}_it{}_nopr{}_ss{}_rs{}.nii'.format(METHOD,NOISE_LEVEL,ITTERATIONS,NUM_PROJECTIONS,SMOOTHING_SIG, RANDOM_SEED)
    nib.save(ni_img, output_dir+output_nifty)
    

    
    # Cleanup.
    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(reconstruction_id)
    astra.data3d.delete(projections_id)
    astra.data3d.delete(phantom_id)


    print('>>> created reconstructions')



if __name__ == "__main__":   
    
    # Configuration.
    input_dir = 'projections'
    output_dir ='test'
    phantom_loc='input/phantom.nii'
    METHODs=['SIRT', 'FBP']
    NOISE_LEVELs=[5e-5, 1e-4, 1e-3 ] 
    RANDOM_SEEDs=[1,2,3,4,5]
    NUM_PROJECTIONSs=[150,250,350]
    ITTERATIONSs=[10]
    SMOOTHING_SIGs=[0]
    
    for method,nl,it,nopr,ss,rs in itertools.product(METHODs,NOISE_LEVELs,ITTERATIONSs,NUM_PROJECTIONSs, SMOOTHING_SIGs, RANDOM_SEEDs):
        print(nl,it,nopr,ss,rs)
        simulation(method,nl,it,nopr,ss,rs, input_dir,output_dir, phantom_loc)
        
    
