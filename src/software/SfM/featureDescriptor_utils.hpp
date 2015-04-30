#include "openMVG/image/image_io.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/features/keypointSet.hpp"
#include "openMVG/image/image_converter.hpp"

/// Feature detector and descriptor interface
#include "nonFree/sift/SIFT.hpp"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include "third_party/progress/progress.hpp"


enum EMethodDetector
{
  eMethodDetector_sift, eMethodDetector_asift
};

/**
 * @brief Compute features and descriptor
 *    - extract sift features and descriptor
 *    - save features and descriptors on disk if it doesn't already exist
 */
template<class DescriptorT, class FeatureT>
void computeFeatDesc(
    const std::vector<std::string>& vec_imageFileNames, const std::string& sOutDir, const size_t detectorType,
    const bool bOctMinus1, const float dPeakThreshold, const size_t coefZoom
  )
{
  using namespace openMVG;

  typedef std::vector<FeatureT> FeatsT;
  typedef std::vector<DescriptorT > DescsT;
  typedef KeypointSet<FeatsT, DescsT > KeypointSetT;

  {
    std::cout << "\n\nEXTRACT FEATURES" << std::endl;

    Image<unsigned char> imageGray;

    C_Progress_display my_progress_bar( vec_imageFileNames.size() );
    for(size_t i=0; i < vec_imageFileNames.size(); ++i)
    {
      KeypointSetT kpSet;

      std::string sFeat = stlplus::create_filespec(sOutDir,
        stlplus::basename_part(vec_imageFileNames[i]), "feat");
      std::string sDesc = stlplus::create_filespec(sOutDir,
        stlplus::basename_part(vec_imageFileNames[i]), "desc");

      if (!stlplus::file_exists(sFeat) || !stlplus::file_exists(sDesc))
      {
        // Not already computed, so compute and save
        ReadImage(vec_imageFileNames[i].c_str(), &imageGray);

        // Compute features and descriptors and export them to file
        if ( detectorType == eMethodDetector_sift )
        {
          SIFTDetector(imageGray,
            kpSet.features(), kpSet.descriptors(),
            bOctMinus1, true, dPeakThreshold);
        }
        else if ( detectorType == eMethodDetector_asift )
        {
            std::cout << "\n\nASIF not implemented." << std::endl;
            exit(-1);
//          ASIFTDetector(imageGray,
//            kpSet.features(), kpSet.descriptors(),
//            coefZoom, dPeakThreshold);
        }
        kpSet.saveToBinFile(sFeat, sDesc);
      }
      ++my_progress_bar;
    }
  }

  }
