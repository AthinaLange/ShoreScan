"""
utils_exif.py

This module provides functions for embedding relevant metadata into images, particularly for
handling camera calibration parameters and extracting metadata from YAML and JSON files.

Dependencies:
    - os
    - json
    - glob
    - yaml
    - datetime
    - pathlib

"""
import os
import json
import glob
import yaml
from datetime import datetime
from pathlib import Path

def loadMultiCamJson(site, camNum, jsonDir, year = None):
    """
    Load JSON metadata for sites with multiple cameras.
    
    :param site: (str) site name
    :param camNum: (str) camera number
    :param jsonDir: (str) Directory containing JSON metadata files.
    :param year: (int, optional) Year to specify a JSON file. Defaults to None.
    
    
    :return dict: JSON metadata tags.
    """
    json_file = f"{jsonDir}/{site}EXIF_c{camNum}_{year}.json" if year else f"{jsonDir}/{site}EXIF_c{camNum}.json"
    with open(json_file) as f:
        json_tags = json.load(f)

    return json_tags

def createCalibDict(site, yamlDir, fileDatetime=None, camNum=None, transect=None):
    """
    Create the dictionary object containing the camera calibration parameters
    (extrinsics, intrinsics, local origin, metadata) for a given site. Read in data
    from YAML files. If necessary, for sites like Madeira Beach, also specify a datetime
    to select the appropriate extrinsic calibration and metadata YAML files. Each dictionary
    also contains descriptions of what each variable is as well as a note describing the dictionary.
    
    :param site: (str) site name
    :param yamlDir: (str) directory containing YAML calibration files.
    :param fileDatetime: (datetime, optional) datetime for a specific file to determine what
                                  YAML file to select for extrinsics and metadata.
    :param camNum: (int, optional) camera number to use when searching for YAML file
    :param transect: (int, optional) transect to use when searching for YAML file
    
    :return calibDict: (dict) dictionary object of variables and their descriptors
    """

    calibDict = {
        "Note": (
            "Intrinsic camera parameters use the Brown distortion model (Brown, D.C., 1971, "
            "'Close-Range Camera Calibration', Photogrammetric Engineering.). Extrinsic "
            "parameters were computed using CIRN Quantitative Coastal Imaging Toolbox."
        )
    }

    extrinsics, extrinsicFile = getYAMLdata(site, yamlDir, fileDatetime=fileDatetime, camNum=camNum, fileSuffix='EO')
    metadata, metadataFile = getYAMLdata(site, yamlDir, fileDatetime=fileDatetime, camNum=camNum, fileSuffix='IO')
    intrinsics, intrinsicFile = getYAMLdata(site, yamlDir, fileDatetime=fileDatetime, camNum=camNum, fileSuffix='metadata')
    
    # Handle transect data if transect is provided
    transectData = {}
    transectFile = None
    if transect:
        transectData, transectFile = getYAMLdata(site, yamlDir=yamlDir, fileDatetime=fileDatetime, camNum=camNum, transectNum=transect, fileSuffix='timestack')
 
    data_fields = {**extrinsics, **intrinsics, **metadata}

    # Read comments from the YAML files
    comment_fields = {
        **readYAMLcomments(extrinsicFile),
        **readYAMLcomments(intrinsicFile),
        **readYAMLcomments(metadataFile),
    }

    # If transect data exists, add its comments
    if transectFile:
        comment_fields.update(readYAMLcomments(transectFile))

    # Read transects into one dictionary
    transect_fields = {key: str(value) for key, value in transectData.items()}

    # Adding everything to calib_dict
    calibDict["data"] = {key: str(value) for key, value in data_fields.items()}
    calibDict["descriptions"] = {key: str(value) for key, value in comment_fields.items()}
    calibDict["transect"] = transect_fields
   
    return calibDict

def getYAMLdata(site, yamlDir, fileSuffix, fileDatetime = None, camNum = None, transectNum=None):
    """
    Generic function to retrieve YAML data for calibration.
    
    :param site: (str) site name
    :param yamlDir: (str) Directory containing YAML files.
    :param fileSuffix: (str) File type suffix (e.g., 'EO', 'IO', 'metadata').
    :param fileDatetime: (datetime, optional): Datetime for selecting appropriate YAML files.
    :param camNum: (int, optional): Camera number.
    
    :return tuple: (dict containing YAML data, filename)
    """
    file_pattern = f"{site}_c{camNum}_*_" if camNum else f"{site}_*_"
    
    
    if fileDatetime:
        if transectNum is None:
            file_pattern = f"{site}_c{camNum}_*_" if camNum else f"{site}_*_"
            yaml_list = glob.glob(os.path.join(yamlDir, f"{file_pattern}{fileSuffix}.yaml"))
            yaml_date_list = uniqueYamlDates(yaml_list)
            index = getClosestPreviousDateIndex(yaml_date_list, fileDatetime)
            closest_date = yaml_date_list[index].strftime("%Y%m%d")
            yaml_file = os.path.join(yamlDir, f"{site}_c{camNum}_{closest_date}_{fileSuffix}.yaml")
        else:
            file_pattern = f"{site}_c{camNum}_{fileSuffix}_*" if camNum else f"{site}_{fileSuffix}_*"
            yaml_list = glob.glob(os.path.join(yamlDir, f"{file_pattern}.yaml"))
            yaml_date_list = uniqueYamlDates(yaml_list)
            index = getClosestPreviousDateIndex(yaml_date_list, fileDatetime)
            closest_date = yaml_date_list[index].strftime("%Y%m%d")
            yaml_file = os.path.join(yamlDir, f"{site}_c{camNum}_{fileSuffix}_{closest_date}_transect{transectNum}.yaml")
    else:
        yaml_file = os.path.join(yamlDir, f"{site}_{fileSuffix}.yaml")
    
    yaml_data = yaml2dict(yaml_file)
    if transectNum:
        yaml_data['transect_date'] = str(closest_date)
    return yaml_data, yaml_file

def getClosestPreviousDateIndex(datetimeList, currentDatetime):
    """
    Compare a list of datetimes to a single (current) datetime and return the index of the closest
    datetime in the list. The closest datetime will be a datetime less than or equal to the currentDatetime.
    Note: All datetimes must be timezone-aware.
    
    :param datetimeList: (datetime) list of datetime (timezone-aware) objects
    :param currentDatetime: (datetime) single datetime (timezone-aware) that each datetime in the
                                     list is compared against
   
    :return closestIndex: (int) index datetime in datetimeList closest to currentDatetime
    """

    # make currentDateTime TZ-naive for comparison
    currentDatetime = currentDatetime.replace(tzinfo=None)
    for k, dNum in enumerate(datetimeList):
        if dNum <= currentDatetime:
            dateDifference = abs(currentDatetime - dNum)

            if k == 0:
                closestIndex = k
                lowestDateDifference = dateDifference
            else:
                # better match for calibration date found
                if dateDifference < lowestDateDifference:
                    lowestDateDifference = dateDifference
                    closestIndex = k

        elif dNum > currentDatetime:
            dateDifference = abs(currentDatetime - dNum)

            if k == 0:
                closestIndex = k
                lowestDateDifference = dateDifference
            else:
                # better match for calibration date found
                if dateDifference < lowestDateDifference:
                    lowestDateDifference = dateDifference
                    closestIndex = k
    return closestIndex

def readYAMLcomments(yamlfile):
    """
    Read a YAML file and add its comments to a dict
    
    :param yamlfile: (str) YAML file to read

    :return comment_dict: (dict) dict of YAML comments
    """
    comment_dict = {}
    with open(yamlfile, "r") as infile:
        lines = infile.readlines()
        for line in lines:
            if line.startswith('#'):
                #clean up comment line and add to dictionary
                line = line.replace('#', '')
                line = line.strip()
                comment_dict[line.split()[0] + '_comment'] = '#' + line
    return comment_dict

def uniqueYamlDates(yamlList):
    """
    Given a list of YAML files, find a list of unique datetimes corresponding to their
    calibration dates.
    
    :param yamlList: (list) list of YAML files

    :return yamlDateList: (list) list of unique datetimes
    """

    yamlDateList = []
    for file in yamlList:
        filename = Path(file).stem
        filename = filename.replace("timestack_", "") if "timestack_" in filename else filename  # Remove 'timestack_' only if it exists
        filenameElements = filename.split("_")

        yamlDateStr = filenameElements[2]
        yamlDate = datetime(
            int(yamlDateStr[0:4]), int(yamlDateStr[4:6]), int(yamlDateStr[6:8])
        )

        if yamlDate not in yamlDateList:
            yamlDateList.append(yamlDate)

    return yamlDateList

def yaml2dict(yamlfile):
    """ Load a YAML file into a dictionary.
    
    :param yamlfile: (str) YAML file to read
    
    :return dictname: (dict) dict interpreted from YAML file
    """
    with open(yamlfile, "r") as infile:
        return yaml.safe_load(infile)
