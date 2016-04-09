
 ban_local_filter v1.0 3D spherical j1 lanczos kernel local interpolation - GNU licensed __chsa__

 The program attempts to locally filter a .mrc density to local resolution given in a local resolution map time efficiently

 Compile with the command:                                     gcc -O3 -lm -lpthread ban_local_filter.c -o ban_local_filter

 Required input: ban_local_filter density_map.mrc resolution_map.mrc output_map.mrc [ --mask (mask.mrc) ] [ --bfac (b-factor) ] [ --oversample ] [ --gauss ]

 -- environmental variable OMP_NUM_THREADS may be set to limit thread count, otherwise all reported processors will be used
 
 -- input map should be b-factor sharpened and unfiltered to avoid degrading high-resolution information by filtering twice
 
 -- resolution maps are truncated at highest reported resolution - without a mask only assigned voxels (<100) are recovered
    the binary mask option forces unassigned voxels within it to be reinterpolated two steps above the max local resolution
 
 -- b-factor dampening may be corrected with the --bfac (b-factor) flag - the b-factor is automatically flipped if negative
    this option currently uses a heuristic but will be updated before publication: please use very cautiously until updated
 
 -- twofold oversampling may be set using --oversample - this allows one to smooth maps close to nyquist for interpretation
    oversampling can be beneficial at high resolution, however not all 3D visualisation software can handle the larger maps
 
 -- the gaussian kernel is provided unless the Lanczos kernel exhibits ringing - techically possible but never yet observed

 -- ResMap & blocres are both supported: no decision on relative quality is made, such judgements are best left to the user
    however it is worth noting that blocres maps run comparatively slowly (several minutes) without the provision of a mask
    ban_loc_filt & blocfilt: ban_loc_filt is faster than blocfilt due to the limited support of the kernel and quantisation

 -- PLEASE NOTE: The output will only ever be as good as your input resmap, map and mask - if they look poor do not proceed
    junk in equals junk out is one thing we will happily guarantee: contact chsaylett@gmail.com for support - best of luck!

 ban_loc_res v1.0 3D spherical j1 lanczos kernel local interpolation - header correct on 16-02-2016 - GNU licensed __chsa__
 [We intend to publish this approach in future, however until then please reference as personal communication Dr CHSAylett]
