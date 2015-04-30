
// Copyright (c) 2014 Bruno DUISIT, Fabien CASTAN, Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "featureDescriptor_utils.hpp"

#include "openMVG/features/features.hpp"
#include "openMVG/matching/matcher_kdtree_flann.hpp"
#include "openMVG/bundle_adjustment/problem_data_container.hpp"
#include "openMVG/bundle_adjustment/pinhole_ceres_functor.hpp"
#include "openMVG/cameras/Camera_IO.hpp"

#include "software/SfMViewer/document.h"
#include "software/SfM/SfMPlyHelper.hpp"
#include "software/SfM/SfMRobust.hpp"

#include "third_party/cmdLine/cmdLine.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <iterator>
#include <iomanip>

using namespace openMVG;
using namespace openMVG::tracks;
using namespace openMVG::bundle_adjustment;

typedef Descriptor<float, 128> DescriptorT;
typedef std::vector<DescriptorT> DescsT;


int main(int argc, char *argv[])
{
  CmdLine cmd;
  std::string sSfMDir;
  std::string sMatchDir;
  std::string sOutDir;
  std::string sImageFilepath;

  bool bOctMinus1 = false;
  size_t coefZoom = 1;
  float dPeakThreshold = 0.04f;
  
  float fDistRatio = .6f;

  cmd.add( make_option('s', sSfMDir, "sfmdir") );
  cmd.add( make_option('m', sMatchDir, "matchdir") );
  cmd.add( make_option('i', sImageFilepath, "image") );
  cmd.add( make_option('o', sOutDir, "outdir") );

  try
  {
      if (argc == 1)
        throw std::string("Invalid command line parameter.");
      cmd.process(argc, argv);
  }
  catch(const std::string& s)
  {
      std::cerr << "Usage: " << argv[0] << ' '
      << "[-m|--matchdir path, the features/descriptors path] "
      << "[-s|--sfmdir path, the SfM_output path] "
      << "[-i|--image path, new image to find 3D pose] "
      << "[-o|--outdir path] "
      << std::endl;

      std::cerr << s << std::endl;
      return EXIT_FAILURE;
  }

  EMethodDetector detectorType = eMethodDetector_sift;

  std::cout << "Load SFM directory" << std::endl;
  Document m_doc;
  if (!m_doc.load(sSfMDir))
  {
    std::cerr << "Can't load SFM directory: " << sSfMDir << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Load descriptors for all images" << std::endl;
  bool bOk = true;
  std::map<size_t, DescsT> map_descPerCameraId;
  // Load the descriptors that correspond to camera names
  for( std::vector<std::string>::const_iterator iName = m_doc._vec_imageNames.begin();
      iName != m_doc._vec_imageNames.end(); ++iName)
  {
    std::string sdescFileName = stlplus::create_filespec(stlplus::folder_append_separator(sMatchDir),
              stlplus::basename_part(*iName) ,"desc");

    std::cout << "Load descriptors for: " << sdescFileName << std::endl;
    size_t index = std::distance<std::vector<std::string>::const_iterator>(m_doc._vec_imageNames.begin(),iName);

    bOk &= loadDescsFromBinFile(
        sdescFileName,
        map_descPerCameraId[index]);
  }

  // Concatenate all sift descriptors corresponding to the 3D structure visibility

  std::cout << "a. count the number of visible 2D points" << std::endl;
  // a. count the number of visible 2D points
  size_t countVisible = 0;
  for (std::map< size_t, submapTrack >::const_iterator iterTracks = m_doc._tracks.begin();
    iterTracks != m_doc._tracks.end();
    ++iterTracks)
  {
    const submapTrack & map_track = iterTracks->second;
    countVisible += map_track.size();
  }
  std::cout << "Number of visible 2D points is " << countVisible << std::endl;
  std::cout << "Number of tracks is " << m_doc._tracks.size() << std::endl;
  std::cout << "map_descPerCameraId.size()" << map_descPerCameraId.size() << std::endl;

  std::vector<DescriptorT> vec_sifts(countVisible);
  size_t siftIndex = 0;  // index inside vec_sifts
  std::vector<size_t> vec_trackSiftIndex(m_doc._tracks.size());  // indexes of tracks inside vec_sifts

  size_t progressStep = m_doc._tracks.size() / 500;
  size_t trackIndex = 0;
  std::vector<DescriptorT>::iterator vec_sifts_it = vec_sifts.begin();
  std::vector<size_t>::iterator vec_trackSiftIndex_it = vec_trackSiftIndex.begin();
  for (std::map< size_t, submapTrack >::const_iterator iterTracks = m_doc._tracks.begin();
        iterTracks != m_doc._tracks.end();
        ++iterTracks)
  {
    if( (trackIndex % progressStep) == 0 )
    {
      std::cout << "Track index: " << trackIndex << std::endl;
    }
    const size_t trackId = iterTracks->first;
    const submapTrack & map_track = iterTracks->second;
//    vec_trackSiftIndex[trackIndex] = siftIndex;
    *vec_trackSiftIndex_it = siftIndex;

    for (submapTrack::const_iterator iterTrack = map_track.begin();
        iterTrack != map_track.end();
        ++iterTrack)
    {
      size_t imaIndex = iterTrack->first;
      size_t featIndex = iterTrack->second;
//      vec_sifts[siftIndex] = map_descPerCameraId[imaIndex][featIndex];
      *vec_sifts_it = map_descPerCameraId[imaIndex][featIndex];
      ++vec_sifts_it;
      ++siftIndex;
    }
    ++vec_trackSiftIndex_it;
    ++trackIndex;
  }

  // Compute feat/desc for the input image

  std::cout << "b. Compute features and descriptors for the new image: " << sImageFilepath << std::endl;
  // b. Compute features and descriptor
  typedef SIOPointFeature FeatureT;
  typedef std::vector<FeatureT> FeatsT;
  typedef KeypointSet<FeatsT, DescsT > KeypointSetT;

  // Compute Feat/desc to files
  std::vector<std::string> vec_imageFileNames;
  vec_imageFileNames.push_back(sImageFilepath);

  computeFeatDesc<DescriptorT, FeatureT>(
    vec_imageFileNames, sMatchDir, detectorType,
    bOctMinus1, dPeakThreshold, coefZoom );

  // Load feat/desc files computed previously
  const std::string sFeatJ = stlplus::create_filespec(
    sMatchDir, stlplus::basename_part(sImageFilepath), "feat");
  const std::string sDescJ = stlplus::create_filespec(
    sMatchDir, stlplus::basename_part(sImageFilepath), "desc");

  std::vector<FeatureT> newImage_feats;
  std::vector<DescriptorT> newImage_descs;

  loadFeatsFromFile(sFeatJ, newImage_feats);
  loadDescsFromBinFile(sDescJ, newImage_descs);

  std::cout << "c. Compute putatives descriptor matches" << std::endl;
  //---------------------------------------
  // c. Compute putatives descriptor matches
  //    - L2 descriptor matching
  //    - Keep correspondences only if NearestNeighbor ratio is ok
  //---------------------------------------
  // Match between descriptors of points cloud and new image descriptors
  std::vector<IndMatch> vec_putativesMatches;
  // Define the matcher and the used metric (Squared L2)
  // ANN matcher could be defined as follow:
  typedef flann::L2<DescriptorT::bin_type> MetricT;
  typedef ArrayMatcher_Kdtree_Flann<DescriptorT::bin_type, MetricT> MatcherT;

  std::vector<int> vec_nIndice;
  {
    std::cout << "PUTATIVE MATCHES" << std::endl;
    
    const int NNN__ = 1; // look for the two nearest distance for each Left descriptor
    std::vector<typename MatcherT::MetricT::ResultType> vec_Distance;
    
    MatcherT matcher;
    DescriptorT::bin_type* siftBuffer = reinterpret_cast<DescriptorT::bin_type*>( &vec_sifts[0] );
    matcher.Build(siftBuffer, vec_sifts.size(), DescriptorT::static_size);
    
    DescriptorT::bin_type* newImage_siftBuffer = reinterpret_cast<DescriptorT::bin_type*>( &newImage_descs[0] );
    matcher.SearchNeighbours(newImage_siftBuffer, newImage_descs.size(), &vec_nIndice, &vec_Distance, NNN__);
  }

  for( size_t i = 0; i < vec_nIndice.size(); ++i )
  {
    vec_putativesMatches.push_back( IndMatch(i, vec_nIndice[i]) );
  }
  Mat pt3D( 3, vec_putativesMatches.size());
  Mat pt2D( 2, vec_putativesMatches.size());
  for( size_t i = 0; i < vec_putativesMatches.size(); ++i )
  {
    const IndMatch& match = vec_putativesMatches[i];
    // Retrieve 3D point index from index in array of all descriptors
    size_t p3d_index = std::distance(
            vec_trackSiftIndex.begin(),
            std::lower_bound(vec_trackSiftIndex.begin(), vec_trackSiftIndex.end(), match._j));
    
    pt3D.col(i) = Map<Vec3f>( & m_doc._vec_points[p3d_index*3] ).cast<double>();
    pt2D.col(i) = newImage_feats[match._i].coords().cast<double>();
  }

  Image<RGBColor> imageRGB;
  ReadImage(sImageFilepath.c_str(), & imageRGB);
  std::pair<size_t,size_t> imageSize(imageRGB.Width(), imageRGB.Height());
  std::vector<size_t> pvec_inliers;

//  Mat3 K;
//  K << 2759, 0, imageRGB.Width()/2,
//       0, 2759, imageRGB.Height()/2, 
//       0, 0, 1;
  Mat34 P;
  double maxError = std::numeric_limits<double>::max();

  std::cout << "Compute the pose of the camera" << std::endl;
  // Compute the pose of the camera
  bool bResection = SfMRobust::robustResection(
    imageSize,
    pt2D,
    pt3D,
    &pvec_inliers,
    NULL, // &K,
    &P,
    &maxError );
  
  std::cout << std::endl
    << "-------------------------------" << std::endl
    << "-- SfM 3D Pose Finder" << std::endl
    << "-- P: " << std::endl << P << std::endl
    << "-- Resection status: " << (bResection ? "OK" : "FAILED") << std::endl
    << "-- max error: " << maxError << std::endl
    << "-- nb detected features in new image: " << newImage_descs.size() << std::endl
    << "-- nb inliers: " << pvec_inliers.size() << std::endl
    << "-------------------------------" << std::endl;
  
  PinholeCamera cam(P); // output cam
  std::cout << cam._K << std::endl;
  
  // save output
  save(stlplus::create_filespec(sOutDir, stlplus::basename_part(sImageFilepath), "noBA.bin"), cam);
  
  //-- Visual export of the cameras poses:
  if(false) // todo: add an option
  {
    std::vector<Vec3> vec_points;
    std::vector<Vec3>  vec_camPos;
    vec_camPos.push_back(cam._C);

    std::vector<Vec3> vec_coloredPoints;
    vec_coloredPoints.push_back(Vec3(255,0,0));

    plyHelper::exportToPly(vec_camPos,
       vec_points,
      stlplus::create_filespec(sOutDir, "poseCam.ply"),
      &vec_coloredPoints);
  }

  std::cout << "Compute Residual error" << std::endl;
  Vec residualErrors(pvec_inliers.size());
  for (size_t i = 0; i < pvec_inliers.size(); ++i)
  {
    Vec3 pt3D_ = pt3D.col(pvec_inliers[i]);
    Vec2 pt2D_ = pt2D.col(pvec_inliers[i]);
    
    residualErrors(i) = cam.Residual(pt3D_, pt2D_);
  }
  std::cout << "Mean Residual error: " << residualErrors.mean() << std::endl;

  std::cout << "Bundle Adjustment" << std::endl;
  // Setup a BA problem
  BA_Problem_data<7> ba_problem;

  // Configure the size of the problem
  ba_problem.num_cameras_ = 1;
  ba_problem.num_points_ = pvec_inliers.size();
  ba_problem.num_observations_ = pvec_inliers.size();

  ba_problem.point_index_.reserve(ba_problem.num_observations_);
  ba_problem.camera_index_.reserve(ba_problem.num_observations_);
  ba_problem.observations_.reserve(2 * ba_problem.num_observations_);

  ba_problem.num_parameters_ = 7 * ba_problem.num_cameras_;
  ba_problem.parameters_.reserve(ba_problem.num_parameters_);

  //double ppx = imageSize.first/2, ppy = imageSize.second/2;
  double ppx = cam._K(0,2), ppy = cam._K(1,2);
  // Fill it with data (tracks and points coords)
  for (int i = 0; i < ba_problem.num_points_; ++i) {
    // Collect the image of point i in each frame.

      ba_problem.camera_index_.push_back(0);
      ba_problem.point_index_.push_back(i);
      const Vec2 & pt = pt2D.col(pvec_inliers[i]);
      ba_problem.observations_.push_back( pt(0) - ppx );
      ba_problem.observations_.push_back( pt(1) - ppy );
  }

  // Add camera parameters (R, t, focal)
  {
    // Rotation matrix to angle axis
    std::vector<double> angleAxis(3);
    ceres::RotationMatrixToAngleAxis((const double*) cam._R.data(), &angleAxis[0]);
    // translation
    Vec3 t = cam._t;
    double focal = cam._K(0,0);
    ba_problem.parameters_.push_back(angleAxis[0]);
    ba_problem.parameters_.push_back(angleAxis[1]);
    ba_problem.parameters_.push_back(angleAxis[2]);
    ba_problem.parameters_.push_back(t[0]);
    ba_problem.parameters_.push_back(t[1]);
    ba_problem.parameters_.push_back(t[2]);
    ba_problem.parameters_.push_back(focal);
  }

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  ceres::Problem problem;
  for (int i = 0; i < ba_problem.num_points_; ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<pinhole_reprojectionError::ErrorFunc_Refine_Camera, 2, 7>(
            new pinhole_reprojectionError::ErrorFunc_Refine_Camera(
                & ba_problem.observations()[2 * i],
                & pt3D.col(pvec_inliers[i])(0)));

    problem.AddResidualBlock(cost_function,
                             NULL, // squared loss
                             ba_problem.mutable_camera_for_observation(0));
  }
  
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  else
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CX_SPARSE))
      options.sparse_linear_algebra_library_type = ceres::CX_SPARSE;
    else
    {
      // No sparse backend for Ceres.
      // Use dense solving
      options.linear_solver_type = ceres::DENSE_SCHUR;
    }
  options.minimizer_progress_to_stdout = false;
  options.logging_type = ceres::SILENT;
#ifdef USE_OPENMP
  options.num_threads = omp_get_num_threads();
#endif // USE_OPENMP

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  
  const double * Rtf = ba_problem.mutable_camera_for_observation(0);
  
  Mat3 R;
  // angle axis to rotation matrix
  ceres::AngleAxisToRotationMatrix(Rtf, R.data());
  Vec3 t(Rtf[3], Rtf[4], Rtf[5]);
  double focal = Rtf[6];
  
  Mat3 KRefined;
  KRefined << focal,0, ppx,
  0, focal, ppy,
  0, 0, 1;
  std::cout << KRefined << std::endl;
  
  PinholeCamera camRefined(KRefined,R,t);

  Vec residualErrorsRefined(pvec_inliers.size());
  Vec residualErrorsRefined2( 2 * pvec_inliers.size() );
  for (size_t i = 0; i < pvec_inliers.size(); ++i)
  {
    Vec3 pt3D_ = pt3D.col(pvec_inliers[i]);
    Vec2 pt2D_ = pt2D.col(pvec_inliers[i]);
    
    Vec2 ptProj = camRefined.Project(pt3D_);
    Vec2 res = ptProj - pt2D_;
    residualErrorsRefined2(i*2) = res(0);
    residualErrorsRefined2(i*2+1) = res(1);
    residualErrorsRefined(i) = camRefined.Residual(pt3D_, pt2D_);
  }
  std::cout << "Mean Residual error after refinement: " << residualErrorsRefined.mean() << std::endl;
  std::cout << "Norm Residual error after refinement: " << residualErrorsRefined2.norm() << std::endl;
  
  // save output
  save(stlplus::create_filespec(sOutDir, stlplus::basename_part(sImageFilepath), "bin"), camRefined);
  
  return( EXIT_FAILURE );
}
