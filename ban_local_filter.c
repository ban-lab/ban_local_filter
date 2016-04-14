/*
 * Copyright 2014-2016 - Dr. Christopher H. S. Aylett
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of version 2 of the GNU General Public License as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details - YOU HAVE BEEN WARNED!
 *
 * BAN_LOC_FILT V1.0: 3D spherical j1 lanczos filter for reinterpolation at local resolution
 *
 * Credit: Chris Aylett { __chsa__ }, Daniel Boehringer, Nenad Ban
 *
 * Author: Chris Aylett { __chsa__ }
 *
 * Date: 16/02/2016
 *
 */

// Library header inclusion for linking
#include <stdio.h>
#include <signal.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>

/* Peak cut at 2 for -0.5 to 0.5 peak, alpha for lobes, true intercept for support,
    max support, twofold oversampling, density truncation, max kernels to store */
#define CUT 2
#define ALPHA 5
#define MAX_RES 50
#define INTERCEPT 4.49340945790906
#define MAX_SUPPORT 1000
#define OVERSAMPLING 2
#define DENSITY_SIGMA_CUT 3
#define MAX_KERNEL_NUMBER 101

typedef struct {
  // Standard MRC header values - crs refer to column, row and segment
  int32_t n_crs[3];
  int32_t mode;
  int32_t start_crs[3];
  int32_t n_xyz[3];
  float   length_xyz[3];
  float   angle_xyz[3];
  int32_t map_crs[3];
  float   d_min;
  float   d_max;
  float   d_mean;
  int32_t ispg;
  int32_t nsymbt;
  int32_t extra[25];
  int32_t ori_xyz[3];
  char    map[4];
  char    machst[4];
  float   rms;
  int32_t nlabl;
  char    label[800];
  // Pointer to be assigned to the map data
  float   *data;
  // Convenience values for map interaction
  float   angpix[3];
  int     stride[3];
} mrc_map;

// External reference to new map to allow the map to be dumped safely should the process be sent SIGTERM
mrc_map *external_reference;

typedef struct {
  // Structure to pass arguments into threads and retrieve their results
  mrc_map       *map;
  mrc_map       *resmap;
  mrc_map       *newmap;
  mrc_map       *mask;
  double        **kernels;
  double        radius_2;
  double        b;
  int           oversample;
  long long int h_start;
  long long int h_step;
} thread_arg;

mrc_map* read_mrc_file(char* filename){
  // Read map header and data and return corresponding data structure
  FILE *f;
  int i;
  f = fopen(filename, "rb");
  if (!f){
    printf("Error reading %s - bad file handle\n", filename);
    exit(1);
  }
  mrc_map *header = calloc(1, sizeof(mrc_map));
  fread(&header->n_crs, 4, 3, f);
  fread(&header->mode, 4, 1, f);
  /* We only accept mode 2 - c float32 / FORTRAN real - because implementing
     checks for other data types would require pointer casting everywhere */
  if (header->mode != 2){
    printf("Error reading %s - not 32 bit data\n", filename);
    return NULL;
  }
  fread(&header->start_crs,  4, 3,   f);
  fread(&header->n_xyz,      4, 3,   f);
  fread(&header->length_xyz, 4, 3,   f);
  fread(&header->angle_xyz,  4, 3,   f);
  fread(&header->map_crs,    4, 3,   f);
  fread(&header->d_min,      4, 1,   f);
  fread(&header->d_max,      4, 1,   f);
  fread(&header->d_mean,     4, 1,   f);
  fread(&header->ispg,       4, 1,   f);
  fread(&header->nsymbt,     4, 1,   f);
  fread(&header->extra,      4, 25,  f);
  fread(&header->ori_xyz,    4, 3,   f);
  fread(&header->map,        1, 4,   f);
  fread(&header->machst,     1, 4,   f);
  fread(&header->rms,        4, 1,   f);
  fread(&header->nlabl,      4, 1,   f);
  fread(&header->label,      1, 800, f);
  /* Assign float array for data, initialize it to zero and read it in - note that the
     endianness is not corrected - architectures may therefore be cross-incompatible*/
  header->data = calloc((header->n_crs[0] * header->n_crs[1] * header->n_crs[2]), sizeof(float));
  if (!header->data){
    return NULL;
  }
  fread(header->data, sizeof(float), (header->n_crs[0] * header->n_crs[1] * header->n_crs[2]), f);
  fclose(f);
  for (i = 0; i < 3; i++){
    header->angpix[i] = header->length_xyz[i] / ((float) header->n_xyz[i]);
  }
  /* These switches set the x, y and z strides for convenience sake - if 
    not available, default parameters are used, but a warning is provided */
  i = 0;
  switch (header->map_crs[0]){
  case 1:
    header->stride[0] = 1;
    break;
  case 2:
    header->stride[1] = 1;
    break;
  case 3:
    header->stride[2] = 1;
    break;
  default:
    i = 1;
  }
  switch (header->map_crs[1]){
  case 1:
    header->stride[0] = header->n_crs[0];
    break;
  case 2:
    header->stride[1] = header->n_crs[0];
    break;
  case 3:
    header->stride[2] = header->n_crs[0];
    break;
  default:
    i = 1;
  }
  switch (header->map_crs[2]){
  case 1:
    header->stride[0] = header->n_crs[0] * header->n_crs[1];
    break;
  case 2:
    header->stride[1] = header->n_crs[0] * header->n_crs[1];
    break;
  case 3:
    header->stride[2] = header->n_crs[0] * header->n_crs[1];
    break;
  default:
    i = 1;
  }
  if (i == 1){
    printf("Error in input - MRC file violates standards - using default parameters\n");
    header->stride[0] = 1;
    header->stride[1] = header->n_crs[0];
    header->stride[2] = header->n_crs[0] * header->n_crs[1];
  }
  return header;
}

mrc_map *new_mrc_from_map(mrc_map *map, int oversample){
  // Create new volume and header and expand as necessary for oversampling
  mrc_map *newmap = calloc(1, sizeof(mrc_map));
  if (!newmap){
    return NULL;
  }
  newmap->n_crs[0] = oversample * map->n_xyz[0];
  newmap->n_crs[1] = oversample * map->n_xyz[1];
  newmap->n_crs[2] = oversample * map->n_xyz[2];

  newmap->n_xyz[0] = oversample * map->n_xyz[0];
  newmap->n_xyz[1] = oversample * map->n_xyz[1];
  newmap->n_xyz[2] = oversample * map->n_xyz[2];

  newmap->map_crs[0] = 1;
  newmap->map_crs[1] = 2;
  newmap->map_crs[2] = 3;

  newmap->mode = 2;

  memcpy(&newmap->length_xyz, &map->length_xyz, 12);
  memcpy(&newmap->angle_xyz,  &map->angle_xyz,  12);
  memcpy(&newmap->ispg,       &map->ispg,        4);
  memcpy(&newmap->nsymbt,     &map->nsymbt,      4);
  memcpy(&newmap->extra,      &map->extra,     100);

  newmap->ori_xyz[0] = (float) oversample * map->ori_xyz[0];
  newmap->ori_xyz[1] = (float) oversample * map->ori_xyz[1];
  newmap->ori_xyz[2] = (float) oversample * map->ori_xyz[2];

  memcpy(&newmap->map,    &map->map,    4);
  memcpy(&newmap->machst, &map->machst, 4);
  memcpy(&newmap->nlabl,  &map->nlabl,  4);

  strncpy(&newmap->label[0],   "\nBAN LOCAL FILTER V1.0 - j1 SPHERICAL LANCZOS INTERPOLATION AT LOCAL RESOLUTION\n", 80);
  strncpy(&newmap->label[80],  "\nC. H. S. AYLETT, D. BOEHRINGER, M. LEIBUNDGUT & N. BAN - IMBB-ETHZ: 16-02-2016\n", 80);

  newmap->angpix[0] = map->angpix[0] / (float) oversample;
  newmap->angpix[1] = map->angpix[1] / (float) oversample;
  newmap->angpix[2] = map->angpix[2] / (float) oversample;

  newmap->stride[0] = 1;
  newmap->stride[1] = newmap->n_crs[0];
  newmap->stride[2] = newmap->n_crs[0] * newmap->n_crs[1];

  newmap->data = calloc(newmap->n_crs[0] * newmap->n_crs[1] * newmap->n_crs[2], sizeof(float));
  if (!newmap->data){
    return NULL;
  }

  return newmap;
}

int write_mrc_file(mrc_map* header, char* filename){
  // Write MRC file given an mrc structure and corresponding data
  int i;

  double total    = header->n_crs[0] * header->n_crs[1] * header->n_crs[2];
  double current  = 0;
  double tmp, sum = 0;

  // Calculate new min, max and mean figures for header
  header->d_min = header->data[0];
  header->d_max = header->data[0];
  for (i = 0; i < total; i++){
    current = header->data[i];
    if (current < header->d_min){
      header->d_min = current;
    }
    if (current > header->d_max){
      header->d_min = current;
    }
    sum += (double) current;
  }
  header->d_mean = (float) (sum / ((double) total));

  // Calculate RMSD to fill in the rms field for scaling
  sum = 0;
  for (i = 0; i < total; i++){
    tmp  = (double) (header->d_mean - header->data[i]);
    sum += tmp * tmp;
  }
  header->rms = (float) sqrt((sum / total));

  // Write out 1024 byte header
  FILE * f;
  f = fopen(filename, "wb");
  if (!f){
    printf("Error writing %s - bad file handle\n", filename);
    return 1;
  }
  fwrite(&header->n_crs,      4, 3,   f);
  fwrite(&header->mode,       4, 1,   f);
  fwrite(&header->start_crs,  4, 3,   f);
  fwrite(&header->n_xyz,      4, 3,   f);
  fwrite(&header->length_xyz, 4, 3,   f);
  fwrite(&header->angle_xyz,  4, 3,   f);
  fwrite(&header->map_crs,    4, 3,   f);
  fwrite(&header->d_min,      4, 1,   f);
  fwrite(&header->d_max,      4, 1,   f);
  fwrite(&header->d_mean,     4, 1,   f);
  fwrite(&header->ispg,       4, 1,   f);
  fwrite(&header->nsymbt,     4, 1,   f);
  fwrite(&header->extra,      4, 25,  f);
  fwrite(&header->ori_xyz,    4, 3,   f);
  fwrite(&header->map,        1, 4,   f);
  fwrite(&header->machst,     1, 4,   f);
  fwrite(&header->rms,        4, 1,   f);
  fwrite(&header->nlabl,      4, 1,   f);
  fwrite(&header->label,      1, 800, f);

  // Write data to file
  fwrite(header->data, 4, total, f);
  fclose(f);

  return 0;
}

double preprocess_maps(mrc_map *resmap, mrc_map *map, mrc_map *mask, int total, double angpix){
  /* Obtain map parameters from resolution map, invert the map, set the
     rms to step size if discrete output, and return exclusion radius_2 */

  int i, count;

  double new =  0;
  double old =  0;
  double max = -1e100;
  double min =  1e100;
  double stp =  1e100;

  // Invert the resmap and handle zeros
  resmap->ispg = 0;
  for (i = 0; i < total; i++){
    new = (double) resmap->data[i];
    if (new > 0){
      resmap->data[i] = 1. / new;
      if ((mask) && (!mask->data[i])){
	continue;
      }
      if ((new > max) && (new < MAX_RES)){
	max = new;
      }
      if (new < min){
	min = new;
      }
      if (((new - old) < stp) && ((new - old) > 0)){
	stp = new - old;
      } else if (((old - new) < stp) && ((old - new) > 0)){
	stp = old - new;
      }
    old = new;
    } else {
      resmap->data[i] = 1 / MAX_RES;
    }
  }

  // Return resmap parameters within header
  resmap->d_min = min;
  resmap->rms   = stp;

  count = (int) round((max - min) / stp) + 1;

  // Step domain is passed back through ISPG
  if (count >= MAX_KERNEL_NUMBER){
    resmap->ispg = 1;
    new = 0;
    old = 0;
    stp = 1e100;
    for (i = 0; i < total; i++){
      new = (double) resmap->data[i];
      if (new > (1 / MAX_RES)){
	if ((mask) && (!mask->data[i])){
	  continue;
	}
	if (((new - old) < stp) && ((new - old) > 0)){
	  stp = new - old;
	} else if (((old - new) < stp) && ((old - new) > 0)){
	  stp = old - new;
	}
      }
      old = new;
    }
    count = (int) round(((1 / min) - (1 / max)) / stp) + 1;
    if (count >= MAX_KERNEL_NUMBER){
      resmap->ispg = 2;
    }
  }

  // Return square radius
  int x, y, z;
  int exclusion_r  = (int) ((min / (CUT * angpix)) * ALPHA * INTERCEPT) / M_PI;
  int stride[3]    = {map->stride[0], map->stride[1], map->stride[2]};
  int map_shape[3] = {map->n_xyz[0], map->n_xyz[1], map->n_xyz[2]};

  double x_c, y_c, z_c;
  double centre[3] = {((double) map_shape[0]) / 2, ((double) map_shape[1]) / 2, ((double) map_shape[2]) / 2};
  double radius_2  = 0;
  double inv_max   = (double) 1 / max;

  // Set masked area to one step above max res if not assigned
  if (mask){
    max = max + resmap->rms;
    if (max > MAX_RES){
      max = MAX_RES;
    }
    resmap->d_max = max;
    inv_max = (double) 1 / max;
    for (i = 0; i < total; i++){
      if (mask->data[i]){
	// Set square radius to max point of mask
	z =  (i / stride[2]);
	y = ((i % stride[2]) / stride[1]);
	x = ((i % stride[2]) % stride[1]);
	z_c = ((double) z) - centre[2];
	y_c = ((double) y) - centre[1];
	x_c = ((double) x) - centre[0];
	if ((z_c * z_c + y_c * y_c + x_c *x_c) > radius_2){
	  radius_2 = (z_c * z_c + y_c * y_c + x_c *x_c);
	}
	// Flush low res data to max within mask
	if (resmap->data[i] < inv_max){
	  resmap->data[i] = inv_max;
	}
      }
    }
  } else {
    if (max > MAX_RES){
      max = MAX_RES;
    }
    resmap->d_max = max;
    inv_max = (double) 1 / max;
    for (i = 0; i < total; i++){
      if (resmap->data[i] >= inv_max){
	// Set square radius to max point of mask
	z =  (i / stride[2]);
	y = ((i % stride[2]) / stride[1]);
	x = ((i % stride[2]) % stride[1]);
	z_c = ((double) z) - centre[2];
	y_c = ((double) y) - centre[1];
	x_c = ((double) x) - centre[0];
	if ((z_c * z_c + y_c * y_c + x_c *x_c) > radius_2){
	  radius_2 = (z_c * z_c + y_c * y_c + x_c *x_c);
	}
      }
    }
  }

  // Pass back step for inverse case
  if (resmap->ispg == 1){
    resmap->rms = stp;
  }

  // Ensure radius doesn't extend beyond map
  if (radius_2 > (centre[0] - exclusion_r) * (centre[0] - exclusion_r)){
    radius_2 = (centre[0] - exclusion_r) * (centre[0] - exclusion_r);
  }
  if (radius_2 > (centre[1] - exclusion_r) * (centre[1] - exclusion_r)){
    radius_2 = (centre[1] - exclusion_r) * (centre[1] - exclusion_r);
  }
  if (radius_2 > (centre[2] - exclusion_r) * (centre[2] - exclusion_r)){
    radius_2 = (centre[2] - exclusion_r) * (centre[2] - exclusion_r);
  }

  return radius_2;
}

int get_thread_number(void){
  // Obtain thread number from environmental variables
  char* thread_number = getenv("OMP_NUM_THREADS");
  int nthreads = 0;
  if (thread_number){
    // If thread number specified by user - use this one
    nthreads = atoi(thread_number);
  }
  if (nthreads < 1){
    // If thread number still not set - try sysconf
    nthreads = sysconf(_SC_NPROCESSORS_ONLN);
  }
  if (nthreads < 1){
  // If variables are both empty - use a single thread
    nthreads = 1;
  }
  return nthreads;
}

float check_identical_angpix(mrc_map* header){
  // Obtain angpix value and check that it is constant in all three directions
  float angpix;
  if ((header->angpix[0] != header->angpix[1]) || (header->angpix[0] != header->angpix[2])){
    printf("Error in inputs - voxels MUST be cubes!\n");
    exit(1);
  }
  angpix = header->angpix[0];
  return angpix;
}

inline double gaussian(double offset_2, double cutoff_2){
  // Spherical Gaussian kernel
  return  pow((2 * M_PI), (-3.0 / 2.0)) * pow(cutoff_2, (-3.0 / 2.0)) * exp(-0.5 * (offset_2 / cutoff_2));
}

inline double lanczos(double offset){
  // Spherical j1-bessel-jinc Lanczos kernel
  if (offset){
    double r  = M_PI * offset;
    double ra = r / ALPHA;
    double j  = (sin(r ) / (r  * r ) - cos(r ) / r ) / r ;
    double ja = (sin(ra) / (ra * ra) - cos(ra) / ra) / ra;
    return (9 * j * ja);
  } else {
    return 1;
  }
}

double *populate_kernel(double resolution, double angpix, double b, int oversample, int gauss){
  // Allocate and return cube populated with b corrected and normalised factors

  // Variations on oversample to save computation
  int    oversample_2   = oversample * oversample;
  int    oversample_3   = oversample * oversample_2;
  double inv_oversample = 1 / (double) oversample;

  // Kernel parameters - cutoff is for lanczos, _2 for gaussian
  double b_corr;
  double cutoff = 1 / (CUT * angpix * resolution);
  double cutoff_2 = 1 / ((2 * CUT * angpix * resolution) * (2 * CUT * angpix * resolution));

  // Update B-factor correction only if utilised
  if (b){
    b_corr = exp((b / (8 * M_PI)) * resolution * resolution);
  } else {
    b_corr = 1;
  }

  // Map bounds to calculate kernel support
  double offset, offset_2;
  double support   = (cutoff * ALPHA * INTERCEPT) / M_PI;
  double support_2 = support * support;

  // Allocate kernel memory and store support
  double *cube = calloc(round((2 * support + 3) * (2 * support + 3) * (2 * support + 3) * oversample_3 + 1), sizeof(double));
  if (!cube){
    return NULL;
  }
  cube[0] = support;

  // Indeces for map, map cube and kernel
  int i, j, k;
  int x, y, z;
  int h;

  // Map cube limits
  int i_0 = (int) (MAX_SUPPORT - support);
  int i_1 = (int) (MAX_SUPPORT + support + 2);
  int j_0 = (int) (MAX_SUPPORT - support);
  int j_1 = (int) (MAX_SUPPORT + support + 2);
  int k_0 = (int) (MAX_SUPPORT - support);
  int k_1 = (int) (MAX_SUPPORT + support + 2);

  // Local centre bounds
  double i_c, j_c, k_c;

  // Loop over map cube
  for (z = 0; z < oversample; z++){
    for (y = 0; y < oversample; y++){
      for (x = 0; x < oversample; x++){

	double integral = 0;
	h = (x % oversample) + oversample * (y % oversample) + oversample_2 * (z % oversample) + 1;

	// Loop over local cube
	for (k = k_0; k < k_1; k++){
	  for (j = j_0; j < j_1; j++){
	    for (i = i_0; i < i_1; i++){

	      // Exclude voxels outside sphere from current
	      i_c = ((double) i) - (MAX_SUPPORT + (((double) x) * inv_oversample));
	      j_c = ((double) j) - (MAX_SUPPORT + (((double) y) * inv_oversample));
	      k_c = ((double) k) - (MAX_SUPPORT + (((double) z) * inv_oversample));
	      offset_2 = (i_c * i_c) + (j_c * j_c) + (k_c * k_c);
	      if (offset_2 > support_2){
		cube[h] = 0;
		h += oversample_3;
		continue;
	      }

	      // Determine new map density
	      if (gauss){
		cube[h] = gaussian(offset_2, cutoff_2);
	      } else {
		offset = sqrt(offset_2) / cutoff;
		cube[h] = lanczos(offset);
	      }
	      integral += cube[h];
	      h += oversample_3;
	    }
	  }
	}
	// Values normalised to unity on a voxel basis, weighted by b correction
	integral = (b_corr * integral) / oversample_3;
	if (integral){
	  for (i = (x % oversample) + oversample * (y % oversample) + oversample_2 * (z % oversample); i < h; i += oversample_3){
	    cube[i + 1] /= integral;
	  }
	}
      }
    }
  }

  return cube;
}

int filter_resmap(thread_arg *args){
  // Filter planar sections of the map using a single thread and steps corresponding to thread number

  /* Specific map bounds and variables - radius from centre then from current point,
        current radius from each - # within indicates + # - _# indicates power */
  int oversample   = args->oversample;
  int oversample_2 = oversample * oversample;
  int oversample_3 = oversample * oversample_2;
  double cent_offset_2, support, suppor2;
  double radius_2       = args->radius_2;
  double radius         = sqrt(radius_2);
  double dbl_oversample = (double) oversample;

  // Local map variables to minimise the number of request collisions slowing progress
  int stride[3]       = {args->map->stride[0], args->map->stride[1], args->map->stride[2]};
  int newstride[3]    = {args->newmap->stride[0], args->newmap->stride[1], args->newmap->stride[2]};
  int map_shape[3]    = {args->map->n_xyz[0], args->map->n_xyz[1], args->map->n_xyz[2]};
  int newmap_shape[3] = {args->newmap->n_xyz[0], args->newmap->n_xyz[1], args->newmap->n_xyz[2]};
  int bar_block       = (newmap_shape[2] >> 6);
  double centre[3]    = {((double) map_shape[0])/2, ((double) map_shape[1])/2, ((double) map_shape[2])/2};

  /* Kernel parameters - map density, resmap resolution, lanczos factor, kernel
       cut-off parameters, low-resolution voxel cut-off, positive B-factor */
  double resolution = 0;
  double density    = 0;
  double old_res    = 0;
  double res_stp    = args->resmap->rms;
  double min_res    = args->resmap->d_min;
  double max_res    = 1 / args->resmap->d_max;
  double *kernel    = args->kernels[0];

  // Map indeces
  int x, y, z, index;
  double dbl_vec[3]     = {0, 0, 0};
  long long int h_start = args->h_start;
  long long int h_step  = args->h_step;

  // Map limits
  int z_0, z_1, y_0, y_1, x_0, x_1;
  x_0 = (int) (centre[0] - radius)     * oversample;
  x_1 = (int) (centre[0] + radius + 2) * oversample;
  y_0 = (int) (centre[1] - radius)     * oversample;
  y_1 = (int) (centre[1] + radius + 2) * oversample;
  z_0 = (int) (centre[2] - radius)     * oversample;
  z_1 = (int) (centre[2] + radius + 2) * oversample;

  // Cube indeces
  int h, i, j, k;

  // Local bounds
  int    i_0, i_1, j_0, j_1, k_0, k_1, step, step_2;
  double x_c, y_c, z_c;

  // Loop over map indices - z steps are by number of threads  each time
  for (z = h_start; z < newmap_shape[2]; z += h_step){

    // Update progress bar
    if (!(z % bar_block)){
      printf("#");
    }

    if (z < z_0 || z > z_1){
      continue;
    }
    for (y = 0; y < newmap_shape[1]; y++){
      if (y < y_0 || y > y_1){
	continue;
      }
      for (x = 0; x < newmap_shape[0]; x++){

	// Make xyz vectors
	if (oversample > 1){
	  dbl_vec[0] = ((double) x) / dbl_oversample;
	  dbl_vec[1] = ((double) y) / dbl_oversample;
	  dbl_vec[2] = ((double) z) / dbl_oversample;
	} else {
	  dbl_vec[0] = ((double) x);
          dbl_vec[1] = ((double) y);
          dbl_vec[2] = ((double) z);
	}

	index = (int) floor(dbl_vec[0]) * stride[0] + floor(dbl_vec[1]) * stride[1] + floor(dbl_vec[2]) * stride[2];

	// Skip voxels outside mask if provided or otherwise if unassigned
	resolution = (double) args->resmap->data[index];
	if (args->mask){
	  if (!args->mask->data[index]){
	    continue;
	  }
	} else if (resolution < max_res){
	  continue;
	}

	// Exclude voxels outside sphere from centre
	x_c = dbl_vec[0] - centre[0];
	y_c = dbl_vec[1] - centre[1];
	z_c = dbl_vec[2] - centre[2];
	cent_offset_2 = (x_c * x_c) + (y_c * y_c) + (z_c * z_c);
	if (cent_offset_2 > radius_2){
	  continue;
	}

	if (resolution != old_res){
	  // Select precalculated cube for product if resolution has changed
	  switch (args->resmap->ispg){
	  case 0:
	    kernel = args->kernels[(int) round(((1 / resolution) - min_res) / res_stp)];
	    break;
	  case 1:
	    kernel = args->kernels[(int) round((resolution - max_res) / res_stp)];
	    break;
	  default:
	    kernel = args->kernels[(int) floor(200 * (resolution - max_res))];
	    break;
	  }
	  support = kernel[0];
	  suppor2 = support + 2;
	  old_res = resolution;
	}

	// Cube limits
	i_0 = (int) (floor(dbl_vec[0]) - support);
	i_1 = (int) (floor(dbl_vec[0]) + suppor2);
	j_0 = (int) (floor(dbl_vec[1]) - support);
	j_1 = (int) (floor(dbl_vec[1]) + suppor2);
	k_0 = (int) (floor(dbl_vec[2]) - support);
	k_1 = (int) (floor(dbl_vec[2]) + suppor2);

	if (oversample > 1){
	  h      = (x % oversample) + oversample * (y % oversample) + oversample_2 * (z % oversample) + 1;
	  index  = (x * newstride[0] + y * newstride[1] + z * newstride[2]);
	  step   = (i_1 - i_0) * oversample_3;
	  step_2 = step * step * oversample_3;
	} else {
	  h      = 1;
	  step   = (i_1 - i_0);
	  step_2 = step * step;
	}

	density = 0;

	// Loop over local cube
	for (k = k_0; k < k_1; k++){
	  if (k < 0 || k >= map_shape[2]){
	    h += step_2;
	    continue;
	  }
	  for (j = j_0; j < j_1; j++){
	    if (j < 0 || j >= map_shape[1]){
	      h += step;
	      continue;
	    }
	    for (i = i_0; i < i_1; i++){
	      if (i < 0 || i >= map_shape[0]){
		h += oversample_3;
		continue;
	      }
	      // Determine new map density
	      density += ((double) args->map->data[i * stride[0] + j * stride[1] + k * stride[2]]) * kernel[h];
	      h       += oversample_3;
	    }
	  }
	}
	// Values returned through arg struct
	if (density){
	  args->newmap->data[index] = (float) density;
	}
      }
    }
  }

  return 0;
}

void bug_out(int signum){
  // Dump map and leave if the threads get stuck
  exit(write_mrc_file(external_reference, "emergency_file_dump.mrc"));
}

int main(int argc, char **argv){

  // Print usage if insufficient arguments provided - capture if required arguments present
  if (argc < 4){
    printf("\n Required input: %s density_map.mrc resolution_map.mrc output_map.mrc [ --mask (mask.mrc) ] [ --bfac (b-factor) ] [ --oversample ] [ --gauss ]\n\n", argv[0]);
    printf(" -- environmental variable OMP_NUM_THREADS may be set to limit thread count, otherwise all reported processors will be used\n");
    printf(" -- input map should be b-factor sharpened but unfiltered to avoid degrading high-resolution information by filtering twice\n");
    printf(" -- resolution maps are truncated at highest reported resolution - without a mask only assigned voxels (<100) are recovered\n");
    printf(" -- the binary mask option forces unassigned voxels within it to be reinterpolated two steps above the max local resolution\n");
    printf(" -- b-factor dampening may be corrected with the --bfac (b-factor) flag - the b-factor is automatically flipped if negative\n");
    printf(" -- this option currently uses a heuristic but will be updated before publication: please use very cautiously until updated\n");
    printf(" -- twofold oversampling may be set using --oversample - this allows one to smooth maps close to nyquist for interpretation\n");
    printf(" -- oversampling can be beneficial at high resolution, however not all 3D visualisation software can handle the larger maps\n");
    printf(" -- the gaussian kernel is provided unless the Lanczos kernel exhibits ringing - techically possible but never yet observed\n\n");
    printf("    ResMap & blocres are both supported: no decision on relative quality is made, such judgements are best left to the user\n");
    printf("    ban_loc_filt & blocfilt: ban_loc_filt is faster than blocfilt due to the limited support of the kernel and quantisation\n\n");
    printf("    PLEASE NOTE: The output will only ever be as good as your input resmap, map and mask - if they look poor do not proceed\n");
    printf("    junk in equals junk out is one thing we will happily guarantee: contact chsaylett@gmail.com for support - best of luck!\n\n");
    printf(" ban_loc_res v1.0 3D spherical j1 lanczos kernel local interpolation - header correct on 16-02-2016 - GNU licensed __chsa__\n");
    printf(" <We intend to publish this approach in future, however until then please reference as personal communication Dr CHSAylett>\n\n");
    return 0;
  }

  printf("\n ban_loc_res v1.0 3D spherical j1 lanczos kernel local interpolation - header correct on 16-02-2016 - GNU licensed __chsa__");
  printf("\n <We intend to publish this approach in future, however until then please reference as personal communication Dr CHSAylett>\n\n");

  int i, j, result_code;

  // Capture user requested settings
  int oversample = 1;
  int gauss      = 0;
  double b = 0;
  int thread_number = get_thread_number();
  mrc_map *mask = NULL;
  printf("\n                 +++ Beginning run with %i threads +++\n", thread_number);
  for (i = 4; i < argc; i++){
    if (!strcmp(argv[i], "--oversample")){
      // Oversample two-fold if requested
      oversample = OVERSAMPLING;
      printf("\n       +++ Oversampling input density %i-fold to smooth edges +++\n", oversample);
    } else if (!strcmp(argv[i], "--gauss")){
      // Use Gaussian kernel if requested
      gauss = 1;
      printf("\n       +++ Using wider 2-sigma gaussian kernel interpolation +++\n");
    } else if (!strcmp(argv[i], "--bfac") && ((i + 1) < argc)){
      // Get B-factor if provided - invert if negative
      b = atof(argv[i + 1]);
      if (b < 0){
	b = -1 * b;
      }
      printf("\n       +++ Setting inverse B-sharpening correction to %6g +++\n", b);
    } else if (!strcmp(argv[i], "--mask") && ((i + 1) < argc)){
      // Use mask if provided - significant speed-up
      mask = read_mrc_file(argv[i + 1]);;
      if (!mask){
	printf("Error while reading mask - exiting\n");
	return 1;
      }
    }
  }

  // Process inputs - 1 map_file, 2 resmap_file, 3 out_map_file
  printf("\n       +++ Loading your density map and resolution map files +++\n");
  mrc_map *map    = read_mrc_file(argv[1]);
  mrc_map *resmap = read_mrc_file(argv[2]);
  if ((!map) || (!resmap)){
    printf("Error while reading maps - exiting\n");
    return 1;
  }

  // Invert resmap, check map similarity and determine resolution step and the required square exclusion radius
  int total = resmap->n_crs[0] * resmap->n_crs[1] * resmap->n_crs[2];
  if (total != (map->n_crs[0] * map->n_crs[1] * map->n_crs[2])){
    printf("Error - differently shaped maps - exiting\n");
    return 1;
  }
  if (mask){
    if (total != (mask->n_crs[0] * mask->n_crs[1] * mask->n_crs[2])){
      printf("Error - differently shaped mask - exiting\n");
      return 1;
    }
  }

  double angpix   = (double) check_identical_angpix(map);
  double radius_2 = preprocess_maps(resmap, map, mask, total, angpix);

  double **kernels = NULL;

  int count;
  switch (resmap->ispg){
  case 0:
    // Use real space if resolution quantised in real space - i.e. Resmap
    printf("\n         +++ Building real-space quantised kernel functions+++\n");
    count   = (int) round((resmap->d_max - resmap->d_min) / resmap->rms) + 1;
    kernels = calloc(count, sizeof(void*));
    for (i = 0; i < count; i++){
      kernels[i] = populate_kernel((double) (1 / (resmap->d_min + ((double) i) * resmap->rms)), angpix, b, oversample, gauss);
      if (!kernels[i]){
	printf("Error during memory allocation for kernels - exiting\n");
	return 1;
      }
    }
    break;
  case 1:
    // Use reciprocal space if resolution quantised in real space - i.e. blocres
    printf("\n      +++ Building reciprocal-space quantised kernel functions+++\n");
    count   = (int) round(((1 / resmap->d_min) - (1 / resmap->d_max)) / resmap->rms) + 1;
    kernels = calloc(count, sizeof(void*));
    for (i = 0; i < count; i++){
      kernels[i] = populate_kernel((double) ((1 / resmap->d_max) + ((double) i) * resmap->rms), angpix, b, oversample, gauss);
      if (!kernels[i]){
	printf("Error during memory allocation for kernels - exiting\n");
	return 1;
      }
    }
    break;
  default:
    // Sample over 100 evenly spread reciprocal resolution kernel functions if continuous
    printf("\n      +++ Building evenly reciprocally spaced kernel functions+++\n");
    count = (int) round((200 / resmap->d_min) - (200 / resmap->d_max)) + 1;
    if (count > MAX_KERNEL_NUMBER){
      printf("Maps extending much beyond nyquist are not supported\n");
      return 1;
    }
    kernels = calloc(count, sizeof(void*));
    for (i = 0; i < count; i++){
      kernels[i] = populate_kernel((double) ((1 / resmap->d_max) + ((double) i) * 0.005), angpix, b, oversample, gauss);
      if (!kernels[i]){
	printf("Error during memory allocation for kernels - exiting\n");
	return 1;
      }
    }
    break;
  }
  

  // Allocate new map
  mrc_map *newmap = new_mrc_from_map(map, oversample);
  if (!newmap){
    printf("Error during memory allocation for new map - exiting\n");
    return 1;
  }
    
  external_reference = newmap;

  // Create progress bar
  int bar_blocks;
  printf("\n       +++ Beginning filtering according to local resolution +++\n\n");
  setbuf(stdout, NULL);
  printf("0%% [");
  if (kernels){
    bar_blocks = (newmap->n_xyz[2] / (newmap->n_xyz[2] >> 6));
  } else {
    bar_blocks = 64;
  }
  for (i = 0; i < bar_blocks; i++){
    printf(" ");
  }
  printf("] 100%%\r0%% [");
  
  // Allocate threads / arguments
  pthread_t      *threads = calloc(thread_number, sizeof(pthread_t));
  thread_arg *thread_args = calloc(thread_number, sizeof(thread_arg));
  if ((!threads) || (!thread_args)){
    printf("Error while initialising threads - exiting\n");
    return 1;
  }

  // Populate thread arguments
  for (i = 0; i < thread_number; i++){
    thread_args[i].map        = map;
    thread_args[i].resmap     = resmap;
    thread_args[i].newmap     = newmap;
    thread_args[i].mask       = mask;
    thread_args[i].kernels    = kernels;
    thread_args[i].radius_2   = radius_2;
    thread_args[i].b          = b;
    thread_args[i].oversample = oversample;
    thread_args[i].h_start    = i;
    thread_args[i].h_step     = thread_number;
  }

  // Create threads
  for (i = 0; i < thread_number; i++){
    result_code = pthread_create(&threads[i], NULL, (void*) filter_resmap, &thread_args[i]);
    if (result_code){
      printf("\nThread initialisation failed!\n");
      return 1;
    }
  }

  struct sigaction action;
  memset(&action, 0, sizeof(struct sigaction));
  action.sa_handler = bug_out;
  sigaction(SIGTERM, &action, NULL);

  // Await thread completion
  for (i = 0; i < thread_number; i++) {
    result_code = pthread_join(threads[i], NULL);
    if (result_code){
      printf("\nThread failed during run!\n");
      return 1;
    }
  }

  // End progress bar
  printf("\n\n");
  printf("                       +++ That's all folks! +++\n\n");
  
  // Write out new map
  return write_mrc_file(newmap, argv[3]);
}
