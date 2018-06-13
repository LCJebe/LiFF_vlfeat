/** @internal
 ** @file     sift.c
 ** @author   Andrea Vedaldi
 ** @brief    Scale Invariant Feature Transform (SIFT) - MEX
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <mexutils.h>
#include <vl/mathop.h>
#include <vl/sift.h>

#include <math.h>
#include <assert.h>

/* option codes */
enum {
  opt_octaves = 0,
  opt_levels,
  opt_first_octave,
  opt_frames,
  opt_edge_thresh,
  opt_peak_thresh,
  opt_norm_thresh,
  opt_magnif,
  opt_window_size,
  opt_orientations,
  opt_float_descriptors,
  opt_verbose
} ;

/* options */
vlmxOption  options [] = {
  {"Octaves",          1,   opt_octaves           },
  {"Levels",           1,   opt_levels            },
  {"FirstOctave",      1,   opt_first_octave      },
  {"Frames",           1,   opt_frames            },
  {"PeakThresh",       1,   opt_peak_thresh       },
  {"EdgeThresh",       1,   opt_edge_thresh       },
  {"NormThresh",       1,   opt_norm_thresh       },
  {"Magnif",           1,   opt_magnif            },
  {"WindowSize",       1,   opt_window_size       },
  {"Orientations",     0,   opt_orientations      },
  {"FloatDescriptors", 0,   opt_float_descriptors },
  {"Verbose",          0,   opt_verbose           },
  {0,                  0,   0                     }
} ;

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Transpose desriptor
 **
 ** @param dst destination buffer.
 ** @param src source buffer.
 **
 ** The function writes to @a dst the transpose of the SIFT descriptor
 ** @a src. The tranpsose is defined as the descriptor that one
 ** obtains from computing the normal descriptor on the transposed
 ** image.
 **/

VL_INLINE void
transpose_descriptor (vl_sift_pix* dst, vl_sift_pix* src)
{
  int const BO = 8 ;  /* number of orientation bins */
  int const BP = 4 ;  /* number of spatial bins     */
  int i, j, t ;

  for (j = 0 ; j < BP ; ++j) {
    int jp = BP - 1 - j ;
    for (i = 0 ; i < BP ; ++i) {
      int o  = BO * i + BP*BO * j  ;
      int op = BO * i + BP*BO * jp ;
      dst [op] = src[o] ;
      for (t = 1 ; t < BO ; ++t)
        dst [BO - t + op] = src [t + o] ;
    }
  }
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Ordering of tuples by increasing scale
 **
 ** @param a tuple.
 ** @param b tuple.
 **
 ** @return @c a[2] < b[2]
 **/

static int
korder (void const* a, void const* b) {
  double x = ((double*) a) [2] - ((double*) b) [2] ;
  if (x < 0) return -1 ;
  if (x > 0) return +1 ;
  return 0 ;
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Check for sorted keypoints
 **
 ** @param keys keypoint list to check
 ** @param nkeys size of the list.
 **
 ** @return 1 if the keypoints are storted.
 **/

vl_bool
check_sorted (double const * keys, vl_size nkeys)
{
  vl_uindex k ;
  for (k = 0 ; k + 1 < nkeys ; ++ k) {
    if (korder(keys, keys + 5) > 0) {
      return VL_FALSE ;
    }
    keys += 5 ;
  }
  return VL_TRUE ;
}

/** ------------------------------------------------------------------
 ** @brief MEX entry point
 **/

void
mexFunction(int nout, mxArray *out[],
            int nin, const mxArray *in[])
{
  enum {IN_I=0,IN_END} ;
  enum {OUT_FRAMES=0, OUT_DOGBUFFINFO, OUT_DOGBUFF, OUT_DESCRIPTORS} ;

  int                verbose = 0 ;
  int                opt ;
  int                next = IN_END ;
  mxArray const     *optarg ;

  vl_sift_pix const *data ;
  int                M, N ;

  int                O     = - 1 ;
  int                S     =   3 ;
  int                o_min =   0 ;

  double             edge_thresh = -1 ;
  double             peak_thresh = -1 ;
  double             norm_thresh = -1 ;
  double             magnif      = -1 ;
  double             window_size = -1 ;

  mxArray           *ikeys_array = 0 ;
  double            *ikeys = 0 ;
  int                nikeys = -1 ;
  vl_bool            force_orientations = 0 ;
  vl_bool            floatDescriptors = 0 ;

  VL_USE_MATLAB_ENV ;

  /* -----------------------------------------------------------------
   *                                               Check the arguments
   * -------------------------------------------------------------- */

  if (nin < 1) {
    mexErrMsgTxt("One argument required.") ;
  } else if (nout > 4) {
    mexErrMsgTxt("Too many output arguments.");
  }

  if (mxGetNumberOfDimensions (in[IN_I]) != 2              ||
      mxGetClassID            (in[IN_I]) != mxSINGLE_CLASS  ) {
    mexErrMsgTxt("I must be a matrix of class SINGLE") ;
  }

  data = (vl_sift_pix*) mxGetData (in[IN_I]) ;
  M    = mxGetM (in[IN_I]) ;
  N    = mxGetN (in[IN_I]) ;

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {

    case opt_verbose :
      ++ verbose ;
      break ;

    case opt_octaves :
      if (!vlmxIsPlainScalar(optarg) || (O = (int) *mxGetPr(optarg)) < 0) {
        mexErrMsgTxt("'Octaves' must be a positive integer.") ;
      }
      break ;

    case opt_levels :
      if (! vlmxIsPlainScalar(optarg) || (S = (int) *mxGetPr(optarg)) < 1) {
        mexErrMsgTxt("'Levels' must be a positive integer.") ;
      }
      break ;

    case opt_first_octave :
      if (!vlmxIsPlainScalar(optarg)) {
        mexErrMsgTxt("'FirstOctave' must be an integer") ;
      }
      o_min = (int) *mxGetPr(optarg) ;
      break ;

    case opt_edge_thresh :
      if (!vlmxIsPlainScalar(optarg) || (edge_thresh = *mxGetPr(optarg)) < 1) {
        mexErrMsgTxt("'EdgeThresh' must be not smaller than 1.") ;
      }
      break ;

    case opt_peak_thresh :
      if (!vlmxIsPlainScalar(optarg) || (peak_thresh = *mxGetPr(optarg)) < 0) {
        mexErrMsgTxt("'PeakThresh' must be a non-negative real.") ;
      }
      break ;

    case opt_norm_thresh :
      if (!vlmxIsPlainScalar(optarg) || (norm_thresh = *mxGetPr(optarg)) < 0) {
        mexErrMsgTxt("'NormThresh' must be a non-negative real.") ;
      }
      break ;

    case opt_magnif :
      if (!vlmxIsPlainScalar(optarg) || (magnif = *mxGetPr(optarg)) < 0) {
        mexErrMsgTxt("'Magnif' must be a non-negative real.") ;
      }
      break ;

    case opt_window_size :
      if (!vlmxIsPlainScalar(optarg) || (window_size = *mxGetPr(optarg)) < 0) {
        mexErrMsgTxt("'WindowSize' must be a non-negative real.") ;
      }
      break ;

    case opt_frames :
      if (!vlmxIsMatrix(optarg, 5, -1)) {
        mexErrMsgTxt("'Frames' must be a 5 x N matrix.") ;
      }
      ikeys_array = mxDuplicateArray (optarg) ;
      nikeys      = mxGetN (optarg) ;
      ikeys       = mxGetPr (ikeys_array) ;
      if (! check_sorted (ikeys, nikeys)) {
        qsort (ikeys, nikeys, 5 * sizeof(double), korder) ;
      } 
      break ;

    case opt_orientations :
      force_orientations = 1 ;
      break ;

    case opt_float_descriptors :
      floatDescriptors = 1 ;
      break ;

    default :
      abort() ;
    }
  }

  /* -----------------------------------------------------------------
   *                                                            Do job
   * -------------------------------------------------------------- */
  {
    VlSiftFilt        *filt ;
    vl_bool            first ;
    double            *frames = 0 ;
    void              *descr  = 0 ;
    int                nframes = 0, reserved = 0, i,j,q ;
	double* OctaveInfoOut = 0;
	float* DogBuffOut = 0;
	int DogBuffW = 0;
	int DogBuffH = 0;
	int DogBuffS = 0;
	int DogBuffO = 0;

    /* create a filter to process the image */
    filt = vl_sift_new (M, N, O, S, o_min) ;


    if (peak_thresh >= 0) vl_sift_set_peak_thresh (filt, peak_thresh) ;
    if (edge_thresh >= 0) vl_sift_set_edge_thresh (filt, edge_thresh) ;
    if (norm_thresh >= 0) vl_sift_set_norm_thresh (filt, norm_thresh) ;
    if (magnif      >= 0) vl_sift_set_magnif      (filt, magnif) ;
    if (window_size >= 0) vl_sift_set_window_size (filt, window_size) ;

    OctaveInfoOut = mxRealloc (OctaveInfoOut, 3 * vl_sift_get_noctaves(filt) * sizeof(double) ) ;  // [width, height, nscales] x noct

	if( nout > 2 )  //  set up to save dog buffer
	{
		DogBuffW = VL_SHIFT_LEFT (M,  -o_min) ;
		DogBuffH = VL_SHIFT_LEFT (N, -o_min) ;
		DogBuffS = (filt->s_max - filt->s_min);
		DogBuffO = vl_sift_get_noctaves(filt);

		int nel = DogBuffW * DogBuffH * DogBuffS * DogBuffO;
		DogBuffOut = mxRealloc (DogBuffOut, nel * sizeof(float) ) ;

		//mexPrintf( "allocated %d elements for output buffer\n", nel );
	}

    if (verbose) {

	  int w   = VL_SHIFT_LEFT (M,  -o_min) ;
	  int h   = VL_SHIFT_LEFT (N, -o_min) ;
	  int nel = w * h ;

      mexPrintf( "vl_sift: s_max: %d; s_min: %d; w: %d; h:%d\n", filt->s_max, filt->s_min, w, h);
	  mexPrintf ("vl_sift: alloc for octave: %d\n", nel * (filt->s_max - filt->s_min + 1) );
	  mexPrintf ("vl_sift: alloc for dog: %d\n",    nel * (filt->s_max - filt->s_min    ) );
	  mexPrintf ("vl_sift: alloc for grad: %d\n",   nel * 2 * (filt->s_max - filt->s_min) );   

      mexPrintf("vl_sift: filter settings:\n") ;
      mexPrintf("vl_sift:   octaves      (O)      = %d\n",
                vl_sift_get_noctaves      (filt)) ;
      mexPrintf("vl_sift:   levels       (S)      = %d\n",
                vl_sift_get_nlevels       (filt)) ;
      mexPrintf("vl_sift:   first octave (o_min)  = %d\n",
                vl_sift_get_octave_first  (filt)) ;
      mexPrintf("vl_sift:   edge thresh           = %g\n",
                vl_sift_get_edge_thresh   (filt)) ;
      mexPrintf("vl_sift:   peak thresh           = %g\n",
                vl_sift_get_peak_thresh   (filt)) ;
      mexPrintf("vl_sift:   norm thresh           = %g\n",
                vl_sift_get_norm_thresh   (filt)) ;
      mexPrintf("vl_sift:   window size           = %g\n",
                vl_sift_get_window_size   (filt)) ;
      mexPrintf("vl_sift:   float descriptor      = %d\n",
                floatDescriptors) ;

      mexPrintf((nikeys >= 0) ?
                "vl_sift: will source frames? yes (%d read)\n" :
                "vl_sift: will source frames? no\n", nikeys) ;
      mexPrintf("vl_sift: will force orientations? %s\n",
                force_orientations ? "yes" : "no") ;
    }

    /* ...............................................................
     *                                             Process each octave
     * ............................................................ */
    i     = 0 ;
    first = 1 ;
    while (1) {
      int                   err ;
      VlSiftKeypoint const *keys  = 0 ;
      int                   nkeys = 0 ;

      if (verbose) {
        mexPrintf ("vl_sift: processing octave %d\n",
                   vl_sift_get_octave_index (filt));
      }

      /* Calculate the GSS for the next octave .................... */
      if (first) {
        err   = vl_sift_process_first_octave (filt, data) ;
        first = 0 ;
      } else {
        err   = vl_sift_process_next_octave  (filt) ;
      }

      if (err) break ;

      if (verbose > 1) {
        mexPrintf("vl_sift: GSS octave %d computed\n",
                  vl_sift_get_octave_index (filt));

        mexPrintf("vl_sift: Octave size : %d x %d\n",
                  vl_sift_get_octave_width(filt), vl_sift_get_octave_height(filt));
      }

      /* Run detector ............................................. */
      if (nikeys < 0) {

		{
			int EntriesPerOct = DogBuffW*DogBuffH*DogBuffS;
			memset( filt->dog, 0, EntriesPerOct*sizeof(float) );  // todo: remove
		}

        vl_sift_detect (filt) ;

		if( !first )
		{
			int CurOctIdx = vl_sift_get_octave_index (filt) - filt->o_min;
			//mexPrintf ("saving to octave: %d\n", CurOctIdx);

			OctaveInfoOut[ CurOctIdx*3 ] = vl_sift_get_octave_width(filt);
			OctaveInfoOut[ CurOctIdx*3 + 1 ] = vl_sift_get_octave_height(filt);
			OctaveInfoOut[ CurOctIdx*3 + 2 ] = DogBuffS;

			if( nout > 2 )  // save dog buffer
			{
				int EntriesPerOct = DogBuffW*DogBuffH*DogBuffS;
				int StartIdx = CurOctIdx * EntriesPerOct;
				DogBuffOut[ StartIdx ] = 2000.0f + CurOctIdx;
				memcpy( &(DogBuffOut[StartIdx]), filt->dog, EntriesPerOct*sizeof(float) );
				//mexPrintf( "entries per oct: %d\n", EntriesPerOct );
			}
		}

        keys  = vl_sift_get_keypoints  (filt) ;
        nkeys = vl_sift_get_nkeypoints (filt) ;
        i     = 0 ;

        if (verbose > 1) {
          printf ("vl_sift: detected %d (unoriented) keypoints\n", nkeys) ;
        }
      } else {
        nkeys = nikeys ;
      }

      /* For each keypoint ........................................ */
      for (; i < nkeys ; ++i) {
        double                angles [4] ;
        int                   nangles ;
        VlSiftKeypoint        ik ;
        VlSiftKeypoint const *k ;

        /* Obtain keypoint orientations ........................... */
        if (nikeys >= 0) {
          vl_sift_keypoint_init (filt, &ik,
                                 ikeys [5 * i + 1] - 1,
                                 ikeys [5 * i + 0] - 1,
                                 ikeys [5 * i + 2],
                                 ikeys [5 * i + 4] ) ;

          if (ik.o != vl_sift_get_octave_index (filt)) {
            break ;
          }

          k = &ik ;

          /* optionally compute orientations too */
          if (force_orientations) {
            nangles = vl_sift_calc_keypoint_orientations
              (filt, angles, k) ;
          } else {
            angles [0] = VL_PI / 2 - ikeys [5 * i + 3] ;
            nangles    = 1 ;
          }
        } else {
          k = keys + i ;
          nangles = vl_sift_calc_keypoint_orientations
            (filt, angles, k) ;
        }

        /* For each orientation ................................... */
        for (q = 0 ; q < nangles ; ++q) {
          vl_sift_pix  buf [128] ;
          vl_sift_pix rbuf [128] ;

          /* compute descriptor (if necessary) */
          if (nout > 3) {  // save descriptors
            vl_sift_calc_keypoint_descriptor (filt, buf, k, angles [q]) ;
            transpose_descriptor (rbuf, buf) ;
          }

          /* make enough room for all these keypoints and more */
          if (reserved < nframes + 1) {
            reserved += 2 * nkeys ;
            frames = mxRealloc (frames, 9 * sizeof(double) * reserved) ;
            if (nout > 3) { // descriptors
              if (! floatDescriptors) {
                descr  = mxRealloc (descr,  128 * sizeof(vl_uint8) * reserved) ;
              } else {
                descr  = mxRealloc (descr,  128 * sizeof(float) * reserved) ;
              }
            }
          }

          /* Save back with MATLAB conventions. Notice tha the input
           * image was the transpose of the actual image. */
          frames [9 * nframes + 0] = k -> y + 1 ;
          frames [9 * nframes + 1] = k -> x + 1 ;
          frames [9 * nframes + 2] = k -> sigma ;
          frames [9 * nframes + 3] = VL_PI / 2 - angles [q] ;
          frames [9 * nframes + 4] = k -> DogVal ;
          frames [9 * nframes + 5] = k -> ix ;
          frames [9 * nframes + 6] = k -> iy ;
          frames [9 * nframes + 7] = k -> is ;
          frames [9 * nframes + 8] = k -> o ;

          if (nout > 3) { // descriptors
            if (! floatDescriptors) {
              for (j = 0 ; j < 128 ; ++j) {
                float x = 512.0F * rbuf [j] ;
                x = (x < 255.0F) ? x : 255.0F ;
                ((vl_uint8*)descr) [128 * nframes + j] = (vl_uint8) x ;
              }
            } else {
              for (j = 0 ; j < 128 ; ++j) {
                float x = 512.0F * rbuf [j] ;
                ((float*)descr) [128 * nframes + j] = x ;
              }
            }
          }

          ++ nframes ;
        } /* next orientation */
      } /* next keypoint */
    } /* next octave */

    if (verbose) {
      mexPrintf ("vl_sift dd: found %d keypoints\n", nframes) ;
    }

    /* ...............................................................
     *                                                       Save back
     * ............................................................ */

    {
      mwSize dims [2] ;

      /* create an empty array */
      dims [0] = 0 ;
      dims [1] = 0 ;
      out[OUT_FRAMES] = mxCreateNumericArray
        (2, dims, mxDOUBLE_CLASS, mxREAL) ;

      /* set array content to be the frames buffer */
      dims [0] = 9 ;
      dims [1] = nframes ;
      mxSetPr         (out[OUT_FRAMES], frames) ;
      mxSetDimensions (out[OUT_FRAMES], dims, 2) ;

      if (nout > 3) { // descriptors
        /* create an empty array */
        dims [0] = 0 ;
        dims [1] = 0 ;
        out[OUT_DESCRIPTORS]= mxCreateNumericArray
          (2, dims,
           floatDescriptors ? mxSINGLE_CLASS : mxUINT8_CLASS,
           mxREAL) ;

        /* set array content to be the descriptors buffer */
        dims [0] = 128 ;
        dims [1] = nframes ;
        mxSetData       (out[OUT_DESCRIPTORS], descr) ;
        mxSetDimensions (out[OUT_DESCRIPTORS], dims, 2) ;
      }

      if (nout > 1) { // dog array info
        /* create an empty array */
        dims [0] = 0 ;
        dims [1] = 0 ;
        out[OUT_DOGBUFFINFO]= mxCreateNumericArray
          (2, dims,
           mxDOUBLE_CLASS,
           mxREAL) ;

        /* set array content to be the descriptors buffer */
        dims [0] = 3 ;
        dims [1] = vl_sift_get_noctaves(filt);
        mxSetData       (out[OUT_DOGBUFFINFO], OctaveInfoOut) ;
        mxSetDimensions (out[OUT_DOGBUFFINFO], dims, 2) ;
      }

      if (nout > 2) { // dog buffer content
        mwSize dims4 [4] ;
        /* create an empty array */
        dims4 [0] = 0 ;
        dims4 [1] = 0 ;
        dims4 [2] = 0 ;
        dims4 [3] = 0 ;
        out[OUT_DOGBUFF]= mxCreateNumericArray
          (4, dims4,
           mxSINGLE_CLASS,
           mxREAL) ;

        /* set array content to be the descriptors buffer */
        dims4 [0] = DogBuffW;
        dims4 [1] = DogBuffH;
        dims4 [2] = DogBuffS;
        dims4 [3] = DogBuffO;
        mxSetData       (out[OUT_DOGBUFF], DogBuffOut) ;
        mxSetDimensions (out[OUT_DOGBUFF], dims4, 4) ;
      }

    }

    /* cleanup */
    vl_sift_delete (filt) ;

    if (ikeys_array)
      mxDestroyArray(ikeys_array) ;

  } /* end: do job */
}
