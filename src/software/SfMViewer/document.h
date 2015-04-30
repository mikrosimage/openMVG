
// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef DOCUMENT
#define DOCUMENT

#include "openMVG/cameras/PinholeCamera.hpp"
#include "openMVG/cameras/Camera_IO.hpp"
#include "openMVG/tracks/tracks.hpp"
using namespace openMVG;

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>

struct Document
{
  std::vector<float> _vec_points;
  std::map<size_t, std::vector<size_t> > _map_visibility; //Inth camera see the Inth 3D point
  tracks::STLMAPTracks _tracks;

  std::map<size_t, PinholeCamera > _map_camera;
  std::vector<std::string> _vec_imageNames;
  std::map<size_t, std::pair<size_t,size_t> > _map_imageSize;

  std::string _sDirectory;


  bool load(const std::string & spath)
  {
    //-- Check if the required file are present.
    _sDirectory = spath;
    std::string sDirectoryPly = stlplus::folder_append_separator(_sDirectory) + "clouds";
    if (stlplus::is_file(stlplus::create_filespec(sDirectoryPly,"visibility","txt"))
      && stlplus::is_file(stlplus::create_filespec(_sDirectory,"views","txt")))
    {
      // Read visibility file (3Dpoint, NbVisbility, [(imageId, featId); ... )
      std::string iFilenamein = stlplus::create_filespec(sDirectoryPly,"visibility","txt");
      std::ifstream iFilein(iFilenamein.c_str());
      std::cout << "Read Visibility: " << iFilenamein << std::endl;
      if (iFilein.is_open())
      {
        size_t trackId = 0;
        while (!iFilein.eof())
        {
          // read one line at a time
          std::string temp;
          std::getline(iFilein, temp);
          std::stringstream sStream(temp);
          float pt[3];
          sStream >> pt[0] >> pt[1] >> pt[2];
          int count;
          sStream >> count;
          size_t imaId, featId;
          for (int i = 0; i < count; ++i)
          {
            sStream >> imaId >> featId;
            _tracks[trackId].insert(std::make_pair(imaId, featId));
            _map_visibility[imaId].push_back(trackId); //imaId camera see the point indexed trackId
          }

          _vec_points.push_back(pt[0]);
          _vec_points.push_back(pt[1]);
          _vec_points.push_back(pt[2]);
          trackId++;
        }
      }
      else
      {
        std::cerr << "Cannot open the visibility file" << std::endl;
      }
    }
    else
    {
      std::cerr << "Required file(s) is missing" << std::endl;
    }

    std::cout << "_tracks.size() " << _tracks.size() << std::endl;
    std::cout << "_map_visibility.size() " << _map_visibility.size() << std::endl;
    std::cout << "_vec_points.size() " << _vec_points.size() << std::endl;

    // Read cameras
    std::string sDirectoryCam = stlplus::folder_append_separator(_sDirectory) + "cameras";
    std::cout << "Read Cameras: " << sDirectoryCam << std::endl;

    size_t camIndex = 0;
    //Read views file
    {
      std::string viewFilename = stlplus::create_filespec(_sDirectory,"views","txt");
      std::cout << "Read views.txt: " << viewFilename << std::endl;
      std::ifstream iFilein(viewFilename.c_str());
      if (iFilein.is_open())
      {
        std::string temp;
        getline(iFilein,temp); //directory name
        getline(iFilein,temp); //directory name
        size_t nbImages;
        iFilein>> nbImages;
        while(iFilein.good())
        {
          getline(iFilein,temp);
          if (!temp.empty())
          {
            std::stringstream sStream(temp);
            std::string sImageName, sCamName;
            size_t w,h;
            float znear, zfar;
            sStream >> sImageName >> w >> h >> sCamName >> znear >> zfar;
            // Read the corresponding camera
            PinholeCamera cam;
            if (!openMVG::load(stlplus::folder_append_separator(sDirectoryCam) + sCamName, cam))
            {
              std::cerr << "Cannot read camera" << std::endl;
              return false;
            }
            _map_camera[camIndex] = cam;

            _vec_imageNames.push_back(sImageName);
            _map_imageSize[camIndex] = std::make_pair(w,h);
            ++camIndex;
          }
          temp.clear();
        }
      }
      std::cout << "_map_camera.size() " << _tracks.size() << std::endl;
      std::cout << "\n Loaded image names : " << std::endl;
      std::copy(_vec_imageNames.begin(), _vec_imageNames.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    }
    return !_map_camera.empty();
  }
};

#endif //DOCUMENT
