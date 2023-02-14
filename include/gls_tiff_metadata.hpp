// Copyright (c) 2021-2022 Glass Imaging Inc.
// Author: Fabio Riccardi <fabio@glass-imaging.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TiffMetadata_hpp
#define TiffMetadata_hpp

#include <string>
#include <variant>
#include <vector>
#include <unordered_map>

#include <tiffio.h>

namespace gls {

typedef std::variant<uint8_t, uint16_t, uint32_t, int8_t, int16_t, int32_t, float, double,
                     std::vector<uint8_t>, std::vector<uint16_t>, std::vector<uint32_t>,
                     std::vector<int8_t>, std::vector<int16_t>, std::vector<int32_t>,
                     std::vector<float>, std::vector<double>, std::string> tiff_metadata_item;

class tiff_metadata: public std::unordered_map<ttag_t, tiff_metadata_item> { };

void readExifMetaData(TIFF* tif, tiff_metadata* metadata);

void readAllTIFFTags(TIFF* tif, tiff_metadata* metadata);

void augment_libtiff_with_custom_tags();

void writeMetadataForTag(TIFF* tif, const tiff_metadata* metadata, ttag_t key);

void writeExifMetadata(TIFF* tif, const tiff_metadata* exif_metadata);

template <typename T>
std::vector<T> getVector(const gls::tiff_metadata& metadata, ttag_t key) {
    const auto& entry = metadata.find(key);
    if (entry != metadata.end()) {
        return std::get<std::vector<T>>(entry->second);
    }
    return std::vector<T>();
}

template <typename T>
bool getValue(const gls::tiff_metadata& metadata, ttag_t key, T* value) {
    const auto& entry = metadata.find(key);
    if (entry != metadata.end()) {
        *value = std::get<T>(entry->second);
        return true;
    }
    return false;
}

// DNG Extension Tags

#define TIFFTAG_DNG_IMAGEWIDTH 61441
#define TIFFTAG_DNG_IMAGEHEIGHT 61442
#define TIFFTAG_DNG_BITSPERSAMPLE 61443

#define TIFFTAG_FORWARDMATRIX1 50964
#define TIFFTAG_FORWARDMATRIX2 50965
#define TIFFTAG_TIMECODES 51043
#define TIFFTAG_FRAMERATE 51044
#define TIFFTAG_REELNAME 51081

#define TIFFTAG_PROFILENAME 50936
#define TIFFTAG_PROFILELOOKTABLEDIMS 50981
#define TIFFTAG_PROFILELOOKTABLEDATA 50982
#define TIFFTAG_PROFILELOOKTABLEENCODING 51108
#define TIFFTAG_DEFAULTUSERCROP 51125

#define TIFFTAG_RATING 18246
#define TIFFTAG_RATINGPERCENT 18249
#define TIFFTAG_TIFFEPSTANDARDID 37398
#define TIFFTAG_DATETIMEORIGINAL 36867

#define TIFFTAG_ISO 34855
#define TIFFTAG_FNUMBER 33437
#define TIFFTAG_EXPOSURETIME 33434
#define TIFFTAG_FOCALLENGHT 37386

#define TIFFTAG_PROFILETONECURVE 50940
#define TIFFTAG_PROFILEEMBEDPOLICY 50941
#define TIFFTAG_ORIGINALDEFAULTFINALSIZE 51089
#define TIFFTAG_ORIGINALBESTQUALITYSIZE 51090
#define TIFFTAG_ORIGINALDEFAULTCROPSIZE 51091
#define TIFFTAG_NEWRAWIMAGEDIGEST 51111

#define TIFFTAG_PREVIEWCOLORSPACE 50970

#define TIFFTAG_ASSHOTPROFILENAME 50934
#define TIFFTAG_PROFILEHUESATMAPDIMS 50937
#define TIFFTAG_PROFILEHUESATMAPDATA1 50938
#define TIFFTAG_PROFILEHUESATMAPDATA2 50939

#define TIFFTAG_OPCODELIST1 51009
#define TIFFTAG_OPCODELIST2 51009
#define TIFFTAG_OPCODELIST3 51022

#define TIFFTAG_NOISEPROFILE 51041
#define TIFFTAG_NOISEREDUCTIONAPPLIED 50935

#define TIFFTAG_IMAGENUMBER 37393

#define TIFFTAG_CAMERACALIBRATIONSIG 50931
#define TIFFTAG_PROFILECALIBRATIONSIG 50932
#define TIFFTAG_PROFILECOPYRIGHT 50942
#define TIFFTAG_PREVIEWAPPLICATIONNAME 50966
#define TIFFTAG_PREVIEWAPPLICATIONVERSION 50967
#define TIFFTAG_PREVIEWSETTINGSDIGEST 50969
#define TIFFTAG_PREVIEWDATETIME 50971

} // namespace gls

#endif /* TiffMetadata_hpp */
