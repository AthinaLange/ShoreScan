def save_to_netcdf(self):
        """
        Saves all available data from the ImageHandler class to a NetCDF file, 
        including the image, metadata, rectification data, and configuration settings.

        :param output_path: (str) Path to save the NetCDF file.
        """
        if self.image is None:
            raise ValueError("No image loaded.")
    
        # Core metadata
        sitePath = self.config.get("site_settingsPath", {}) or utils_CIRN.prompt_for_directory("Select the site metadata settings JSON file")
        if not sitePath.endswith('.json'):
            sitePath = os.path.join(sitePath, "site_settings.json")  # Append 'site_settings.json' if it's a directory

        with open(sitePath, "r") as file:
            site = json.load(file)
        site = site[self.site]
                
        onlyDate = self.datetime.strftime('%Y-%m-%d')

        # Handle sample frequency conversion
        if site["sampling"]["collection_unit"].lower() in ['hz', 'hertz']:
            sample_frequency_Hz = site["sampling"]["sample_frequency"]
        elif site["sampling"]["collection_unit"].lower() in ['s', 'sec', 'seconds']:
            sample_frequency_Hz = 1 / site["sampling"]["sample_frequency"]
        else:
            raise ValueError("Invalid collection_unit. Expected 'Hz', 'Hertz', 's', 'sec', or 'seconds'.")

        # Handle sample period length conversion
        if site["sampling"]["sample_period_unit"].lower() in ['s', 'sec', 'seconds']:
            sample_period_length = site["sampling"]["sample_period_length"] / 60  # Convert seconds to minutes
        elif site["sampling"]["sample_period_unit"].lower() in ['m', 'min', 'minutes']:
            sample_period_length = site["sampling"]["sample_period_length"]  # Already in minutes
        else:
            raise ValueError("Invalid sample_period_unit. Expected 's', 'sec', 'seconds', 'm', 'min', or 'minutes'.")
        utmzone = site["siteInfo"]["utmZone"]

        global_attrs = {
                "name": self.image_name,
                "conventions": "CF-1.6",
                "institution": "U.S. Geological Survey",
                "source": "Mounted camera image capture",
                "references": site["siteInfo"]["references"],
                "metadata_link": getattr(site["siteInfo"], "metadata_link", ""),
                "title": f"{site['siteInfo']['siteLocation']} {self.datetime} UTC: CoastCam image",
                "program": "Coastal-Marine Hazards and Resources",
                "project": "Next Generation Total Water Level and Coastal Change Forecasts",
                "contributors": site["siteInfo"]["contributors"],
                "year": self.datetime.year,
                "date": onlyDate,
                "site_location": site["siteInfo"]["siteLocation"],
                "description": f"Coastcam {self.image_type} image sampled at {sample_frequency_Hz:.2f} Hz for {sample_period_length:.2f} minutes at {self.site}. Collection began at {self.datetime} UTC.",
                "sample_period_length": f"{sample_period_length:.2f} minutes",
                "data_origin": site["siteInfo"]["dataOrigin"],
                "coord_system": "UTM",
                "utm_zone": site["siteInfo"]["utmZone"],
                "cam_make": site["siteInfo"]["camMake"],
                "cam_model": site["siteInfo"]["camModel"],
                "cam_lens": site["siteInfo"]["camLens"],
                "data_type": "imagery",
                "local_timezone": site["siteInfo"]["timezone"],
                "verticalDatum":  site['siteInfo']['verticalDatum'],
                "verticalDatum_description": "North America Vertical Datum of 1988 (NAVD 88)",
                "freqLimits": site["sampling"]["freqLimits"],
                "freqLimits_description": "Frequency limits on sea-swell and infragravity wave band (Hz). [SS upper, SS/IG, IG lower]",
            
                "datetime": str(self.datetime),
                "site": self.site,
                "camera": self.camera,
                "image_path": str(self.image_path),
                "image_type": self.image_type,
                "intrinsics": json.dumps(self.metadata["intrinsics"]),
                "intrinsics_description": "NU, NV, coU, coV, fx, fy, d1, d2, d3, t1, t2",
                "extrinsics": json.dumps(self.metadata["extrinsics"]),
                "intrinsics_description": "Eastings, Northings, Elevation, Azimuth, Tilt, Roll (degrees)"
            }
            
        image_arr = np.array(self.image, dtype=np.uint8)
        
        dims = {
            "U_dim": image_arr.shape[0],
            "V_dim": image_arr.shape[1],
            "Color_dim": 3
        }

        products_attrs = {
                    "I":{
                        "long_name": 'image pixel color value',
                        "color_band":'RGB',
                        "description":'8-bit image color values of the image. Three dimensions: pixel rows, pixel columns, color band, where the colors are RGB--R(ed) color band equals 0, G(reen) color band equals 1, and B(blue) color band equals 2.',
                        "coordinates": 'U, V'
                    },
                    "crs_utm":{
                        "grid_mapping_name" : 'transverse_mercator',
                        "scale_factor_at_central_meridian" : 0.999600,
                        "longitude_of_central_meridian" : -177 + (int(site["siteInfo"]["utmZone"][0:2]) - 1) * 6,
                        "latitude_of_projection_origin" : 0.000000,
                        "false_easting" : 500000.000000,
                        "false_northing" : 0.000000
                    },
                    "crs_latlon":{
                        "grid_mapping_name": 'latitude_longitude'
                    }
        }
        
        data_vars={
                "I": (["U_dim", "V_dim", "Color_dim"], image_arr, products_attrs["I"]),
                "crs_utm": ([], 0, products_attrs["crs_utm"]),
                "crs_latlon": ([], 0, products_attrs["crs_latlon"])
        }
        
        coords={
                "U_dim": ("U_dim", np.arange(dims["U_dim"])),
                "V_dim": ("V_dim", np.arange(dims["V_dim"])),
                "Color_dim": ("Color_dim", np.arange(dims["Color_dim"]))
        }
            
        rectified_image = self.processing_results.get("rectified_image", {})
        shoreline = self.processing_results.get("shoreline", {})
        runup = self.processing_results.get("runup", {})

        if rectified_image:
            global_attrs.update({"origin_easting": rectified_image.get("Origin_Easting", np.nan),
                                "origin_northing": rectified_image.get("Origin_Northing", np.nan),
                                "origin_UTMZone": rectified_image.get("Origin_UTMZone", np.nan),
                                "origin_angle": rectified_image.get("Origin_Angle", np.nan),
                                "origin_angle_units": "degrees",
                                "dx": rectified_image.get("dx", np.nan),
                                "dy": rectified_image.get("dy", np.nan),
                                "dx_dy_units": "meters",
                                "tide": rectified_image.get("tide", np.nan),
                                "tide_description": f"tide level used in projection, {site['siteInfo']['verticalDatum']}m"
            })
            
            dims.update({"X_dim": np.shape(np.array(rectified_image.get("localX", [])))[0],
                        "Y_dim": np.shape(np.array(rectified_image.get("localY", [])))[1],
            })
            
            products_attrs.update({
                    "Ir":{
                        "long_name": 'rectified image pixel color value',
                        "color_band":'RGB',
                        "description":'Rectified image. Three dimensions: X, Y, color band, where the colors are RGB--R(ed) color band equals 0, G(reen) color band equals 1, and B(blue) color band equals 2.',
                        "coordinates": 'X,Y', 
                        "grid_mapping": 'crs_utm crs_wgs84 crs_local'
                        },
                    "localX":{
                        "long_name": "Local cross-shore coordinates in meters of rectified image.",
                        "min_value": np.around(np.array(rectified_image.get("localX", [])).min(), decimals=3),
                        "max_value": np.around(np.array(rectified_image.get("localX", [])).min(), decimals=3),
                        "units": "meters",
                        "description":" Local cross-shore coordinates in meters of rectified image. Rotated based on shorenormal angle and origin. ",
                        },
                    "localY":{
                        "long_name": "Local along-shore coordinates in meters of rectified image.",
                        "min_value": np.around(np.array(rectified_image.get("localY", [])).min(), decimals=3),
                        "max_value": np.around(np.array(rectified_image.get("localY", [])).min(), decimals=3),
                        "units": "meters",
                        "description":" Local along-shore coordinates in meters of rectified image. Rotated based on shorenormal angle and origin. ",
                    },
                    "Eastings":{
                        "long_name" : f"Universal Transverse Mercator Zone {utmzone} Easting coordinate of rectified image",
                        "units" : 'meters',
                        "min_value" : np.around(np.array(rectified_image.get("Eastings", [])).min(), decimals=3),
                        "max_value" : np.around(np.array(rectified_image.get("Eastings", [])).max(), decimals=3),
                        "description" : f'Cross-shore coordinates of data in the rectified image projected onto the beach surface at {self.site}. Described using UTM Zone {site["siteInfo"]["utmZone"]} Easting in meters.',
                    },
                    "Northings":{
                        "long_name" : f"Universal Transverse Mercator Zone {utmzone} Northing coordinate of rectified image",
                        "units" : 'meters',
                        "min_value" : np.around(np.array(rectified_image.get("Northings", [])).min(), decimals=3),
                        "max_value" : np.around(np.array(rectified_image.get("Northings", [])).max(), decimals=3),
                        "description" : f'Cross-shore coordinates of data in the rectified image projected onto the beach surface at {self.site}. Described using UTM Zone {site["siteInfo"]["utmZone"]} Northing in meters.',
                    },
                    "Z":{
                        "long_name" : 'elevation',
                        "units" : 'meters',
                        "coordinates": 'X,Y',
                        "description" : f'Elevation (z-value) in {site["siteInfo"]["verticalDatum"]} of rectified image projected onto the beach surface at {self.site}.',
                        "datum": site["siteInfo"]["verticalDatum_description"],
                        "min_value" : np.around(np.nanmin(np.array(rectified_image.get("Z", []))), decimals=3),
                        "max_value" : np.around(np.nanmax(np.array(rectified_image.get("Z", []))), decimals=3)
                    }
            })

            coords.update({"X_dim": ("X_dim", np.arange(dims["X_dim"])),
                           "Y_dim": ("Y_dim", np.arange(dims["Y_dim"]))
            })

            data_vars.update({"Ir": (["X_dim", "Y_dim", "Color_dim"], np.array(rectified_image.get("Ir", []), dtype=np.uint8), products_attrs["Ir"]),
                            "localX": (["X_dim", "Y_dim"], np.around(np.array(rectified_image.get("localX", [])), decimals=3), products_attrs["localX"]),
                            "localY": (["X_dim", "Y_dim"], np.around(np.array(rectified_image.get("localY", [])), decimals=3), products_attrs["localY"]),
                            "Eastings": (["X_dim", "Y_dim"], np.around(np.array(rectified_image.get("Eastings", [])), decimals=3), products_attrs["Eastings"]),
                            "Northings": (["X_dim", "Y_dim"], np.around(np.array(rectified_image.get("Northings", [])), decimals=3), products_attrs["Northings"]),
                            "Z": (["X_dim", "Y_dim"], np.around(np.array(rectified_image.get("Z", [])), decimals=3), products_attrs["Z"]),
            })
    
        if shoreline: 
            global_attrs.update({"tide": shoreline.get("tide", 0),
                                "tide_description": f"tide level used in projection, {site['siteInfo']['verticalDatum']}m",
                                "SAM_model": shoreline.get("model",{})
            })
            
            dims.update({"X_dim": np.shape(np.array(rectified_image.get("localX", [])))[0],
                        "Y_dim": np.shape(np.array(rectified_image.get("localY", [])))[1],
                        "XYZ_dim": 3,
                        "UV_dim": 2
            })
           
            coords.update({"X_dim": ("X_dim", np.arange(dims["X_dim"])),
                           "Y_dim": ("Y_dim", np.arange(dims["Y_dim"]))
            })

            products_attrs.update({
                    "shoreline_rectified_coords":{
                        "long_name": 'UTM coordinates of shoreline from oblique image.',
                        "description":'Rectified shoreline coordinates on UTM using calculated EO and IO. Projected to given tide level.',
                        "coordinates": 'X, Y, Z',
                        "grid_mapping": 'crs_utm crs_wgs84 crs_local'
                    },
                    "shoreline_pixel_coords":{
                        "long_name": '[U,V] coordinates of the shoreline from the oblique image.',
                        "description":'Shoreline as determined by the median of the SAM and large errors with the watershed algorithm are removed.',
                        "coordinates": 'U, V', 
                        "grid_mapping": 'crs_utm crs_wgs84 crs_local'
                        },
                    "bottom_boundary":{
                        "long_name": "Segment anything determined shoreline",
                        "units": "pixels",
                        "description":"First 3 columns: 3 attemps at determining the shoreline from segment anything. 4th column: Vertical median of 3 shorelines. Used to compare with watershed algorithm shoreline and basis for final shoreline. Markers for ocean initialized from either: bright) random points spread across the largest, brightest continous area or timex) points at least 10 pixels above the matching brightest shoreline or point (1000,300).",
                        },
                    "watershed_coords":{
                        "long_name": "[U,V] coordinates of the shoreline as determined by a watershed algorithm.",
                        "units": "pixels",
                        "coordinates": 'U, V',
                        "description":" Markers for ocean initialized from points at least 10 pixels above the median SAM bottom boundary, markers for sand from points at least 10 pixels below.",
                    },
                    "y_distance":{
                        "long_name" : "Vertical distance in pixels between watershed_coords and median bottom boundary (SAM). ",
                        "units" : 'pixels',
                        "min_value" : np.around(np.array(shoreline.get("y_distance", [])).min(), decimals=3),
                        "max_value" : np.around(np.array(shoreline.get("y_distance", [])).max(), decimals=3),
                        "description" :  'Any horizontal locations that have a y_distance > 30 pixels are excluded from the final shoreline. Helps with stability of the estimates. '
                    },
                    "rmse_value":{
                        "long_name" : "Mean rmse of vertical distance in pixels between watershed_coords and median bottom boundary (SAM).",
                        "units" : 'pixels',
                        "min_value" : np.around(np.array(shoreline.get("rmse_value", [])).min(), decimals=3),
                        "max_value" : np.around(np.array(shoreline.get("rmse_value", [])).max(), decimals=3),
                        "description" : "Provides general estimate of error of shoreline prediction.",
                    }
            })

            data_vars.update({"shoreline_rectified_coords": (["U_dim", "XYZ_dim"], np.array(shoreline.get("rectified_shoreline",[])), products_attrs["shoreline_rectified_coords"]),
                              "shoreline_pixel_coords": (["U_dim", "UV_dim"], np.array(shoreline.get("shoreline_coords",[])), products_attrs["shoreline_pixel_coords"]),
                              "bottom_boundary_attemps": (["U_dim", 4], np.array(shoreline.get("bottom_boundary",[])), products_attrs["bottom_boundary"]),
                              "watershed_coords": (["U_dim", "UV_dim"], np.array(shoreline.get("watershed_coords",[])), products_attrs["watershed_coords"]),
                              "y_distance": (["U_dim"], np.array(shoreline.get("y_distance",[])), products_attrs["y_distance"]),
                              "rmse_value": ([], np.array(shoreline.get("rmse_value",[])), products_attrs["rmse_value"])
            })  
            
        if runup:
            TWLstats = runup.get('TWLstats')
            verticalDatum = site['siteInfo']['verticalDatum']
            freqLimits = site['sampling']['freqLimits']

            t_sec = np.around(np.arange(0, sample_period_length*60, 1/sample_frequency_Hz)[:self.image.shape[0]], decimals=3) # timeseries in seconds
            T = np.array([self.datetime + timedelta(seconds = t) for t in t_sec])

            global_attrs.update({"origin_easting": runup.get("Origin_Easting", np.nan),
                                "origin_northing": runup.get("Origin_Northing", np.nan),
                                "origin_UTMZone": runup.get("Origin_UTMZone", np.nan),
                                "origin_angle": runup.get("Origin_Angle", np.nan),
                                "origin_angle_units": "degrees",
                                "tide": runup.get("tide", np.nan),
                                "tide_description": f"tide level used in projection, {site['siteInfo']['verticalDatum']}m",
                                "transect_date_definition": runup.get("transect_date_definition"),
                                "transect_date_description": "date where U,V coordinates were defined based on recent EO."
            })
            
            dims.update({"X_dim": np.shape(self.image)[0],
                        "T_dim": np.shape(self.image)[1],
                        "TWLstats_dim": runup.get('TWLstats').get('S', np.array([])).size
            })

            coords.update({"X_dim": ("X_dim", np.arange(dims["X_dim"])),
                           "T_dim": T
            })

            products_attrs.update({
                                "I":{
                                    "long_name": 'image pixel color value',
                                    "color_band":'RGB',
                                    "description":'8-bit image color values of the timestack. Three dimensions: time, spatial axis, color band, where the colors are RGB--R(ed) color band equals 0, G(reen) color band equals 1, and B(blue) color band equals 2. The horizontal axis of the image is the spatial axis. The different crs_ mappings represent the same coordinates in UTM, local, and longitude/latitude.',
                                    "coordinates": "time x_utm",
                                    "grid_mapping": 'crs_utm crs_wgs84 crs_local'
                                },
                                "Ri":{
                                    "long_name": "pixel coordinate of runup",
                                    "description": "pixel coordinate of wave runup line as found by the segformer model.",
                                    "units": "pixels"
                                },
                                "runup_val":{
                                    "description": "Runup value used to extract runup from softmax score."
                                },
                                "rundown_val":{
                                    "description": "Rundown value used to extract rundown from softmax score."
                                },
                                "U_transect": {
                                    "long_name": "pixel coordinate along the horizontal axis of the image where timestack was sampled",
                                    "min_value": float(np.min(runup.get("U",[]))),
                                    "max_value": float(np.max(runup.get("U",[]))),
                                    "units": "pixel",
                                    "description": f"Pixel coordinate along the horizontal axis (cross-shore) of the image where timestack was sampled at {self.site}. Obtained from image collection beginning {transect_date}",
                                },
                                "V_transect": {
                                    "long_name": "pixel coordinate along the vertical axis of the image where timestack was sampled",
                                    "min_value": float(np.min(runup.get("V",[]))),
                                    "max_value": float(np.max(runup.get("V",[]))),
                                    "units": "pixel",
                                    "description": f"Pixel coordinate along the vertical axis (time) of the image where timestack was sampled at {self.site}. Obtained from image collection beginning {transect_date}",
                                },
                                "T": {
                                "standard_name": "time",
                                    "long_name": "datetime",
                                    "format": "YYYY-MM-DD HH:mm:SS+00:00",
                                    "time_zone": "UTC",
                                    "description": "Times that pixels were sampled to create the timestack. The dimension length is the number of samples in the timestack. Each sample has a time value represented as a datetime.",
                                    "sample_freq": f"{sample_frequency_Hz} Hertz",
                                    "sample_length_interval": f"{sample_period_length*60} seconds",
                                    "min_value": T[0].isoformat(),
                                    "max_value": T[-1].isoformat(),
                                },
                                "Eastings":{
                                    "long_name" : f"Universal Transverse Mercator Zone {utmZone} Easting coordinate of cross-shore timestack pixels",
                                    "units" : 'meters',
                                    "min_value" : np.around(np.min(runup.get("Eastings",[])), decimals=3),
                                    "max_value" : np.around(np.max(runup.get("Eastings",[])), decimals=3),
                                    "description" : f'Eastings coordinates of data in the timestack pixels projected onto the beach surface at {self.site}. Described using UTM Zone {utmZone} Easting in meters.',
                                },
                                "Northings":{
                                    "long_name" : f'Universal Transverse Mercator Zone {utmZone} Northing coordinate of cross-shore timestack pixels',
                                    "units" : 'meters',
                                    "min_value" : np.around(np.min(runup.get("Northings",[])), decimals=3),
                                    "max_value" : np.around(np.max(runup.get("Northings",[])), decimals=3),
                                    "description" : f'Northings coordinates of data in the timestack pixels projected onto the beach surface at {self.site}. Described using UTM Zone {utmZone} Northing in meters.',
                                },
                                "Z":{
                                    "long_name" : 'Elevation',
                                    "units" : f"{site['siteInfo']['verticalDatum']}m",
                                    "description" : f'Elevation (z-value) in {verticalDatum} of timestack pixels projected onto the beach surface at {self.site}.',
                                    "datum" :site["siteInfo"]["verticalDatum"],
                                    "min_value" : np.around(np.nanmin(runup.get("Z",[])), decimals=3),
                                    "max_value" : np.around(np.nanmax(runup.get("Z",[])), decimals=3)
                                },
                                "DEM":{
                                    "description": "Digital Elevation Model used for runup projection.",
                                    "units": "meters"
                                },
                                "Hrunup":{
                                    "long_name": f"[X, Y] coordinates of wave runup in Eastings, Northings for UTM zone {runup.get("Origin_UTMZone", np.nan)}.",
                                    "min_value": [np.min(runup.get('Hrunup',0)[:,0]), np.min(runup.get('Hrunup',0)[:,1])],
                                    "max_value": [np.max(runup.get('Hrunup',0)[:,0]), np.max(runup.get('Hrunup',0)[:,1])],
                                    "units": "meters ",
                                    "description": "Horizontal coordinates of wave runup as converted from U,V coordinates to Eastings, Northings from EO/IO.",
                                    "coordinates": "X,Y"
                                },
                                "TWL":{
                                    "long_name": "Total water level elevation timeseries",
                                    "min_value": float(runup.get('Zrunup',0).min()),
                                    "max_value": float(runup.get('Zrunup',0).max()),
                                    "units": f"{site['siteInfo']['verticalDatum']}m",
                                    "description": "Vertical elevation of wave runup (total water level).",
                                }    
                        })
            
            TWL_attrs = {
                            "2exceedence_peaksVar":{
                                "long_name": "2 percent exceedence value for twl peaks",
                                "units": "meters"
                            },
                            "2exceedence_notpeaksVar":{
                                "long_name": "2 percent exceedence value for twl timeseries",
                                "units": "meters"
                            },
                            "meanVar":{
                                "long_name": "mean TWL",
                                "units": "meters",
                                "description" : "offshore mean twl + wave setup"
                            },
                            "TpeakVar":{
                                "long_name": "peak swash period",
                                "units": "seconds",
                                "description": "runup (or twl) timeseries peak period"
                            },
                            "TmeanVar":{
                                "long_name": "mean swash period",
                                "units": "seconds",
                                "description": "runup (or twl) timeseries mean period"
                            },
                            "SsigVar":{
                                "long_name": "significant swash",
                                "units":"meters"
                            },
                            "Ssig_SSVar":{
                                "long_name": "significant swash in sea-swell (SS) band",
                                "units": "meters",
                                "description": f"Between {freqLimits[0]} Hz and {freqLimits[1]} Hz."
                            },
                            "Ssig_IGVar":{
                                "long_name": "significant swash in infragravity (IG) band",
                                "units": "meters",
                                "description": f"Between {freqLimits[1]} Hz and {freqLimits[2]} Hz."
                            },
                            "SpectrumVar":{
                                "long_name": "runup (or twl) spectral density array",
                                "units": "meters^2/Hertz"
                            },
                            "FrequencyVar":{
                                "long_name": "runup (or twl) frequency array",
                                "units": "Hertz"
                            }
                }

            data_vars.update({"I": (["U_dim", "V_dim", "Color_dim"], image_arr, products_attrs["I"]),
                            "Ri": (["T_dim"], np.array(runup.get("Ri",[])), products_attrs["Ri"]),
                            "runup_val": ([], np.array(runup.get("runup_val",[])), products_attrs["runup_val"]),
                            "rundown_val": ([], np.array(runup.get("rundown_val",[])), products_attrs["rundown_val"]),
                            "U_transect": (["X_dim"], np.array(runup.get("U",[])), products_attrs["U_transect"]),
                            "V_transect": (["X_dim"], np.array(runup.get("V",[])), products_attrs["V_transect"]),
                            "Eastings": (["X_dim"], np.array(runup.get("Eastings",[])), products_attrs["Eastings"]),
                            "Northings": (["X_dim"], np.array(runup.get("Northings",[])), products_attrs["Northings"]),
                            "Z": (["X_dim"], np.array(runup.get("Z",[])), products_attrs["Z"]),
                            "T": (["T_dim"], np.array(T), products_attrs["T"]),
                            "DEM": ([], np.array(runup.get("DEM",[])), products_attrs["DEM"]),
                            "Hrunup": (["T_dim", 2], np.array(runup.get("Hrunup",[])), products_attrs["Hrunup"]),
                            "TWL": (["T_dim"], np.array(runup.get("Zrunup",[])), products_attrs["TWL"]),
                            "TWLstats_2exceedence_peaks":([], TWLstats.get('R2', None), TWL_attrs["2exceedence_peaksVar"]),
                            "TWLstats_2exceedence_notpeaks":([], TWLstats.get('eta2', None), TWL_attrs["2exceedence_notpeaksVar"]),
                            "TWLstats_mean":([], TWLstats.get('setup', None), TWL_attrs["meanVar"]),
                            "TWLstats_Tpeak":([], TWLstats.get('Tp', None), TWL_attrs["TpeakVar"]),
                            "TWLstats_Tmean":([], TWLstats.get('Ts', None), TWL_attrs["TmeanVar"]),
                            "TWLstats_Ssig":([], TWLstats.get('Ss', None), TWL_attrs["SsigVar"]),
                            "TWLstats_Ssig_SS":([], TWLstats.get('Ssin', None), TWL_attrs["Ssig_SSVar"]),
                            "TWLstats_Ssig_IG":([], TWLstats.get('Ssig', None), TWL_attrs["Ssig_IGVar"]),
                            "TWLstats_spectrum":(["TWLstats_dim"], np.around(TWLstats.get('S', np.array([])), decimals=6), TWL_attrs["SpectrumVar"]),
                            "TWLstats_frequency":(["TWLstats_dim"], np.around(TWLstats.get('f', np.array([])), decimals=4), TWL_attrs["FrequencyVar"]),
            })  
        
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = coords,
            attrs = global_attrs
        )
      
        # Save to NetCDF
        saveDir = self.config.get("netcdfDir", os.path.join(os.getcwd(), 'netcdf'))
        os.makedirs(saveDir, exist_ok = True)
        saveName = os.path.join(saveDir, self.image_name + ".nc")
        ds.to_netcdf(saveName)
        print(f"All data saved to {saveName}")

