// Copyright (c) 2021-2023 Glass Imaging Inc.
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

#include "gls_tiff_metadata.hpp"

#include <iostream>

#include "gls_logging.h"

static const char* TAG = "DEMOSAIC";

// #define DEBUG_TIFF_TAGS 1

namespace gls {

struct TiffFieldInfo {
    const ttag_t field_tag;        /* field's tag */
    const int field_readcount;     /* read count/TIFF_VARIABLE/TIFF_SPP */
    const TIFFDataType field_type; /* type of associated data */
    const char* field_name;        /* ASCII name */
};

std::string getFieldName(const TIFFField* tf) {
    const auto field_name = TIFFFieldName(tf);
    return (field_name != nullptr) ? std::string(field_name) : std::to_string(TIFFFieldTag(tf));
}

template <class T>
bool readMetaDataItem(TIFF* tif, const TIFFField* tf, tiff_metadata* metadata) {
    const auto field_tag = TIFFFieldTag(tf);
    const auto field_readcount = TIFFFieldReadCount(tf);

    if (field_readcount == TIFF_VARIABLE2 || field_readcount == TIFF_VARIABLE || field_readcount > 1) {
        size_t count = 0;
        T* data;

        if (field_readcount == TIFF_VARIABLE) {
            uint16_t gotcount = 0;
            TIFFGetField(tif, field_tag, &gotcount, &data);
            count = gotcount;
        } else if (field_readcount == TIFF_VARIABLE2) {
            uint32_t gotcount = 0;
            TIFFGetField(tif, field_tag, &gotcount, &data);
            count = gotcount;
        } else {
            TIFFGetField(tif, field_tag, &data);
            count = field_readcount;
        }

        std::vector<T> values;
        values.resize(count);
        for (unsigned i = 0; i < count; i++) {
            values[i] = data[i];
        }
#ifdef DEBUG_TIFF_TAGS
        std::cout << "New metadata vector (" << values.size() << ") " << getFieldName(tf) << ": ";
        for (int i = 0; i < values.size() && i < 10; i++) {
            const auto& v = values[i];
            if (sizeof(v) == 1) {
                std::cout << (int)v;
            } else {
                std::cout << v;
            }
            if (i < 9 && i < values.size() - 1) {
                std::cout << ", ";
            } else if (i == 9 && i < values.size() - 1) {
                std::cout << "...";
            }
        }
        std::cout << std::endl;
#endif
        metadata->insert({field_tag, values});
        return true;
    } else if (field_readcount == 1) {
        T data;
        TIFFGetField(tif, field_tag, &data);
#ifdef DEBUG_TIFF_TAGS
        std::cout << "New metadata scalar " << getFieldName(tf) << ": " << data << std::endl;
#endif
        metadata->insert({field_tag, data});
        return true;
    }
    return false;
}

bool readMetaDataString(TIFF* tif, const TIFFField* tf, tiff_metadata* metadata) {
    const auto field_tag = TIFFFieldTag(tf);
    const auto field_readcount = TIFFFieldReadCount(tf);

    const char* data;

    if (field_readcount == TIFF_VARIABLE2) {
        // Happens with undefined ASCII tags which are defaulted to TIFF_VARIABLE2 by libtiff
        uint32_t gotcount = 0;
        TIFFGetField(tif, field_tag, &gotcount, &data);
    } else if (field_readcount == TIFF_VARIABLE || field_readcount > 0) {
        TIFFGetField(tif, field_tag, &data);
    }

#ifdef DEBUG_TIFF_TAGS
    std::cout << "New metadata string " << getFieldName(tf) << ": " << data << std::endl;
#endif
    metadata->insert({field_tag, data});

    return true;
}

void readMetadataForTag(TIFF* tif, tiff_metadata* metadata, ttag_t field_tag) {
    const TIFFField* tf = TIFFFieldWithTag(tif, field_tag);

    const auto field_type = TIFFFieldDataType(tf);
    switch (field_type) {
        case TIFF_BYTE: {
            readMetaDataItem<uint8_t>(tif, tf, metadata);
            break;
        }
        case TIFF_UNDEFINED: {
            readMetaDataItem<uint8_t>(tif, tf, metadata);
            break;
        }
        case TIFF_ASCII: {
            readMetaDataString(tif, tf, metadata);
            break;
        }
        case TIFF_SHORT: {
            readMetaDataItem<uint16_t>(tif, tf, metadata);
            break;
        }
        case TIFF_LONG: {
            readMetaDataItem<uint32_t>(tif, tf, metadata);
            break;
        }
        case TIFF_SBYTE: {
            readMetaDataItem<int8_t>(tif, tf, metadata);
            break;
        }
        case TIFF_SSHORT: {
            readMetaDataItem<int16_t>(tif, tf, metadata);
            break;
        }
        case TIFF_SLONG: {
            readMetaDataItem<int32_t>(tif, tf, metadata);
            break;
        }
        case TIFF_SRATIONAL:
        case TIFF_RATIONAL:
        case TIFF_FLOAT: {
            readMetaDataItem<float>(tif, tf, metadata);
            break;
        }
        case TIFF_DOUBLE: {
            readMetaDataItem<double>(tif, tf, metadata);
            break;
        }
        case TIFF_IFD:
        case TIFF_IFD8:
#ifdef DEBUG_TIFF_TAGS
            std::cout << "Skipping offset field: " << getFieldName(tf) << std::endl;
#endif
            break;
        default:
            throw std::runtime_error("Unknown TIFF field type: " + std::to_string(field_type));
    }
}

void readAllTIFFTags(TIFF* tif, tiff_metadata* metadata) {
    if (tif) {
        int tag_count = TIFFGetTagListCount(tif);
        for (int i = 0; i < tag_count; i++) {
            ttag_t field_tag = TIFFGetTagListEntry(tif, i);

            readMetadataForTag(tif, metadata, field_tag);
        }
    }
}

void readExifMetaData(TIFF* tif, tiff_metadata* exif_metadata) {
    if (tif) {
        // Go back to the first (main) directory
        TIFFSetDirectory(tif, 0);

        // Go to the EXIF directory
        toff_t exif_offset;
        if (TIFFGetField(tif, TIFFTAG_EXIFIFD, &exif_offset)) {
            gls::logging::LogDebug(TAG) << "Reading EXIF metadata..." << std::endl;
            TIFFReadEXIFDirectory(tif, exif_offset);

            readAllTIFFTags(tif, exif_metadata);

            gls::logging::LogDebug(TAG) << "Read " << exif_metadata->size() << " EXIF metadata entries." << std::endl;
        }
    }
}

void writeExifMetadata(TIFF* tif, const tiff_metadata* exif_metadata) {
    TIFFWriteDirectory(tif);

    if (TIFFCreateEXIFDirectory(tif) != 0) {
        std::cerr << "TIFFCreateEXIFDirectory() failed." << std::endl;
    } else {
        gls::logging::LogDebug(TAG) << "Saving " << exif_metadata->size() << " EXIF metadata entries." << std::endl;
        for (auto entry : *exif_metadata) {
            writeMetadataForTag(tif, exif_metadata, entry.first);
        }

        unsigned char exifVersion[4] = {'0', '2', '3', '1'}; /* EXIF 2.31 version is 4 characters of a string! */
        if (!TIFFSetField(tif, EXIFTAG_EXIFVERSION, exifVersion)) {
            throw std::runtime_error("Can't write EXIFTAG_EXIFVERSION.");
        }

        uint64_t dir_offset_EXIF = 0;
        if (!TIFFWriteCustomDirectory(tif, &dir_offset_EXIF)) {
            throw std::runtime_error("TIFFWriteCustomDirectory() with EXIF failed.");
        }

        TIFFSetDirectory(tif, 0);
        TIFFSetField(tif, TIFFTAG_EXIFIFD, dir_offset_EXIF);

        // Write directory to file
        TIFFWriteDirectory(tif);
    }
}

template <typename T>
void writeMetadataItem(TIFF* tif, const TIFFField* tf, const tiff_metadata_item& item) {
    const auto writeCount = TIFFFieldWriteCount(tf);
    uint32_t tag = TIFFFieldTag(tf);
    if (writeCount == 1) {
        const auto value = std::get<T>(item);
#ifdef DEBUG_TIFF_TAGS
        std::cout << "Saving metadata scalar " << getFieldName(tf) << ": " << value << std::endl;
#endif
        if (!TIFFSetField(tif, tag, value)) {
            std::cerr << "Can't write TIFF tag " << tag << " with value " << value << std::endl;
        }
    } else {
        const auto values = std::get<std::vector<T>>(item);
#ifdef DEBUG_TIFF_TAGS
        std::cout << "Saving metadata vector (" << values.size() << ") " << getFieldName(tf) << ": ";
        for (int i = 0; i < values.size() && i < 10; i++) {
            const auto& v = values[i];
            if (sizeof(v) == 1) {
                std::cout << (int)v;
            } else {
                std::cout << v;
            }
            if (i < 9 && i < values.size() - 1) {
                std::cout << ", ";
            } else if (i == 9 && i < values.size() - 1) {
                std::cout << "...";
            }
        }
        std::cout << std::endl;
#endif
        if (writeCount < 0) {
            if (!TIFFSetField(tif, tag, (uint16_t)values.size(), values.data())) {
                std::cerr << "Can't write TIFF tag " << tag << std::endl;
            }
        } else {
            if (writeCount != values.size()) {
                throw std::runtime_error("Vector size mismatch, should be: " + std::to_string(writeCount) +
                                         ", got: " + std::to_string(values.size()));
            }
            if (!TIFFSetField(tif, tag, values.data())) {
                std::cerr << "Can't write tag " << tag << std::endl;
            }
        }
    }
}

void writeMetadataString(TIFF* tif, const TIFFField* tf, const tiff_metadata_item& item) {
    const auto string = std::get<std::string>(item);
#ifdef DEBUG_TIFF_TAGS
    std::cout << "Saving metadata string " << getFieldName(tf) << ": " << string << std::endl;
#endif
    uint32_t tag = TIFFFieldTag(tf);
    if (!TIFFSetField(tif, tag, string.c_str())) {
        std::cerr << "Can't write tag " << tag << " with data " << string << std::endl;
    }
}

void writeMetadataForTag(TIFF* tif, const tiff_metadata* metadata, ttag_t tag) {
    const auto entry = metadata->find(tag);
    if (entry != metadata->end()) {
        const TIFFField* tf = TIFFFieldWithTag(tif, tag);
        if (tf) {
            const auto field_type = TIFFFieldDataType(tf);

            switch (field_type) {
                case TIFF_BYTE: {
                    writeMetadataItem<uint8_t>(tif, tf, entry->second);
                    break;
                }
                case TIFF_UNDEFINED: {
                    writeMetadataItem<uint8_t>(tif, tf, entry->second);
                    break;
                }
                case TIFF_ASCII: {
                    writeMetadataString(tif, tf, entry->second);
                    break;
                }
                case TIFF_SHORT: {
                    writeMetadataItem<uint16_t>(tif, tf, entry->second);
                    break;
                }
                case TIFF_LONG: {
                    writeMetadataItem<uint32_t>(tif, tf, entry->second);
                    break;
                }
                case TIFF_SBYTE: {
                    writeMetadataItem<int8_t>(tif, tf, entry->second);
                    break;
                }
                case TIFF_SSHORT: {
                    writeMetadataItem<int16_t>(tif, tf, entry->second);
                    break;
                }
                case TIFF_SLONG: {
                    writeMetadataItem<int32_t>(tif, tf, entry->second);
                    break;
                }
                case TIFF_SRATIONAL:
                case TIFF_RATIONAL:
                case TIFF_FLOAT: {
                    writeMetadataItem<float>(tif, tf, entry->second);
                    break;
                }
                case TIFF_DOUBLE: {
                    writeMetadataItem<double>(tif, tf, entry->second);
                    break;
                }
                default:
                    throw std::runtime_error("Unknown TIFF field type: " + std::to_string(field_type));
            }
        }
    }
}

// DNG Extension Tags

static const TIFFFieldInfo xtiffFieldInfo[] = {
    {TIFFTAG_FORWARDMATRIX1, -1, -1, TIFF_SRATIONAL, FIELD_CUSTOM, 1, 1, (char*)"ForwardMatrix1"},
    {TIFFTAG_FORWARDMATRIX2, -1, -1, TIFF_SRATIONAL, FIELD_CUSTOM, 1, 1, (char*)"ForwardMatrix2"},

    {TIFFTAG_PROFILENAME, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"ProfileName"},
    {TIFFTAG_PROFILELOOKTABLEDIMS, 3, 3, TIFF_LONG, FIELD_CUSTOM, 1, 0, (char*)"ProfileLookTableDims"},
    {TIFFTAG_PROFILELOOKTABLEDATA, -1, -1, TIFF_FLOAT, FIELD_CUSTOM, 1, 1, (char*)"ProfileLookTableData"},
    {TIFFTAG_PROFILELOOKTABLEENCODING, 1, 1, TIFF_LONG, FIELD_CUSTOM, 1, 0, (char*)"ProfileLookTableEncoding"},
    {TIFFTAG_DEFAULTUSERCROP, 4, 4, TIFF_RATIONAL, FIELD_CUSTOM, 1, 0, (char*)"DefaultUserCrop"},

    {TIFFTAG_RATING, 1, 1, TIFF_SHORT, FIELD_CUSTOM, 1, 0, (char*)"Rating"},
    {TIFFTAG_RATINGPERCENT, 1, 1, TIFF_SHORT, FIELD_CUSTOM, 1, 0, (char*)"RatingPercent"},
    {TIFFTAG_TIFFEPSTANDARDID, -1, -1, TIFF_BYTE, FIELD_CUSTOM, 1, 1, (char*)"TIFF-EP Standard ID"},

    {TIFFTAG_ISO, -1, -1, TIFF_SHORT, FIELD_CUSTOM, 1, 1, (char*)"ISO"},
    {TIFFTAG_FNUMBER, -1, -1, TIFF_FLOAT, FIELD_CUSTOM, 1, 1, (char*)"FNumber"},
    {TIFFTAG_EXPOSURETIME, -1, -1, TIFF_FLOAT, FIELD_CUSTOM, 1, 1, (char*)"ExposureTime"},
    {TIFFTAG_FOCALLENGHT, -1, -1, TIFF_FLOAT, FIELD_CUSTOM, 1, 1, (char*)"FocalLength"},
    {TIFFTAG_DATETIMEORIGINAL, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"DateTimeOriginal"},

    {TIFFTAG_PROFILETONECURVE, -1, -1, TIFF_FLOAT, FIELD_CUSTOM, 1, 1, (char*)"ProfileToneCurve"},
    {TIFFTAG_PROFILEEMBEDPOLICY, 1, 1, TIFF_LONG, FIELD_CUSTOM, 1, 0, (char*)"ProfileEmbedPolicy"},
    {TIFFTAG_ORIGINALDEFAULTFINALSIZE, 2, 2, TIFF_LONG, FIELD_CUSTOM, 1, 0, (char*)"OriginalDefaultFinalSize"},
    {TIFFTAG_ORIGINALBESTQUALITYSIZE, 2, 2, TIFF_LONG, FIELD_CUSTOM, 1, 0, (char*)"OriginalBestQualitySize"},
    {TIFFTAG_ORIGINALDEFAULTCROPSIZE, 2, 2, TIFF_RATIONAL, FIELD_CUSTOM, 1, 0, (char*)"OriginalDefaultCropSize"},
    {TIFFTAG_NEWRAWIMAGEDIGEST, 16, 16, TIFF_BYTE, FIELD_CUSTOM, 1, 0, (char*)"NewRawImageDigest"},

    {TIFFTAG_PREVIEWCOLORSPACE, 1, 1, TIFF_LONG, FIELD_CUSTOM, 1, 0, (char*)"PreviewColorSpace"},

    {TIFFTAG_ASSHOTPROFILENAME, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"AsShotProfileName"},
    {TIFFTAG_PROFILEHUESATMAPDIMS, 3, 3, TIFF_LONG, FIELD_CUSTOM, 1, 0, (char*)"ProfileHueSatMapDims"},
    {TIFFTAG_PROFILEHUESATMAPDATA1, -1, -1, TIFF_FLOAT, FIELD_CUSTOM, 1, 1, (char*)"ProfileHueSatMapData1"},
    {TIFFTAG_PROFILEHUESATMAPDATA2, -1, -1, TIFF_FLOAT, FIELD_CUSTOM, 1, 1, (char*)"ProfileHueSatMapData2"},

    {TIFFTAG_NOISEPROFILE, 2, 2, TIFF_FLOAT, FIELD_CUSTOM, 1, 0, (char*)"NoiseProfile"},
    {TIFFTAG_NOISEREDUCTIONAPPLIED, 1, 1, TIFF_RATIONAL, FIELD_CUSTOM, 1, 0, (char*)"NoiseReductionApplied"},

    {TIFFTAG_IMAGENUMBER, 1, 1, TIFF_LONG, FIELD_CUSTOM, 1, 0, (char*)"ImageNumber"},

    {TIFFTAG_CAMERACALIBRATIONSIG, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"CameraCalibrationSig"},
    {TIFFTAG_PROFILECALIBRATIONSIG, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"ProfileCalibrationSig"},
    {TIFFTAG_PROFILECOPYRIGHT, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"ProfileCopyright"},
    {TIFFTAG_PREVIEWAPPLICATIONNAME, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"PreviewApplicationName"},
    {TIFFTAG_PREVIEWAPPLICATIONVERSION, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"PreviewApplicationVersion"},
    {TIFFTAG_PREVIEWSETTINGSDIGEST, -1, -1, TIFF_BYTE, FIELD_CUSTOM, 1, 1, (char*)"PreviewSettingsDigest"},
    {TIFFTAG_PREVIEWDATETIME, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"PreviewDateTime"},
};

static TIFFExtendProc parent_extender = NULL;  // In case we want a chain of extensions

static void registerCustomTIFFTags(TIFF* tif) {
    // Install the extended Tag field info
    TIFFMergeFieldInfo(tif, xtiffFieldInfo, sizeof(xtiffFieldInfo) / sizeof(xtiffFieldInfo[0]));

    if (parent_extender) parent_extender(tif);
}

void augment_libtiff_with_custom_tags() {
    static bool first_time = true;
    if (first_time) {
        parent_extender = TIFFSetTagExtender(registerCustomTIFFTags);
        first_time = false;
    }
}

}  // namespace gls
