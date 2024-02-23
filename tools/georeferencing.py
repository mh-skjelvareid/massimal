import math
import numpy as np
import crs_tools


def pushbroom_width_on_ground(opening_angle_deg,relative_altitude):
    """ Calculate width of pushbroom camera footprint on ground 
    
    # Input arguments:
    opening_angle_deg:
        Opening angle of camera in degrees
    relative_altitude:
        Altitude of camera relative to ground

    Returns:
    width:
        Width of pushbroom camera image on ground
        Units are the same as the units used for relative altitude.

    # Notes:
    - Assumes that camera is perfectly "centered" (no roll), 
    and that ground is perfectly flat. 
    """
    return 2*math.tan(math.radians(opening_angle_deg/2))*relative_altitude


def world_file_from_lcf_times_files(lcf_file_path,times_file_path,world_file_path,
                                    n_crosstrack_pixels=900,camera_opening_angle_deg=36.5,
                                    altitude_offset=0):
    """ Calculate world file (affine transformation) for raster image based on .lcf and .times files 
    
    # Input parameters:
    lcf_file_path:
        Path to .lcf file, containing camera IMU and GNSS information
        It is assumed that 
        - Column 1 contains times in units of seconds
        - Column 4 contains longitude in (decimal) degrees
        - Column 5 contains latitude in (decimal) degrees
        - Column 6 contains altitude in meters relative to WGS-54 ellipsoid
    times_file_path:
        Path to .times file. It is assumed that has a single column,
        containing timestamps for every line for the image.
    world_file_path:
        Path for output world file. Should have file extension corresponding to
        image format, e.g. 'pgw' for PNG files, or, alternatively,
        just 'wld' (accepted by GDAL and QGIS). 
         
    # Keyword parameters:
    n_crosstrack_pixels:
        Number of pixels "cross-track" (also called number of "samples")
    camera_opening_angle_deg:
        Opening angle of pushbroom camera in degrees.
    altitude_offset:
        Offset between altitude as measured by camera in LCF file, and actual altitude.
        Example: If LCF file altitude was 30 meters, and actual altitude was 35 meters,
        set altitude_offset = 5 m. Actual altitude is calculated as 
        mean LCF altitude + altitude_offset.

    # Notes:
    The "world file" parameters are written to 
        6-element tuple containing parameters for "world file" affine transformation
        world_file_parameters = (A, D, B, E, C, F), where
        A: x-component of the pixel width (x-scale)
        D: y-component of the pixel width (y-skew)
        B: x-component of the pixel height (x-skew)
        E: y-component of the pixel height (y-scale)
        C: x-coordinate of the center of image's upper left pixel 
        F: y-coordinate of the center of image's upper left pixel
        All values are in the units of the map coordinate system (here: UTM)

        The parameters are part of an affine transformation given by
            x = A*ix + B*iy + C
            y = E*ix + E*iy + F
        where ix and iy are the column and row indices of the raster image, respectively.
    """

    # Load LCF data
    lcf_data = np.loadtxt(lcf_file_path)
    lcf_times = lcf_data[:,0] - lcf_data[0,0] # Use time relative to first timestamp
    lcf_long = lcf_data[:,4]
    lcf_lat = lcf_data[:,5]
    lcf_alt = lcf_data[:,6]
    # TODO: Include pitch and roll data from LCF

    # Load image .times data (time for each image line)
    image_times = np.loadtxt(times_file_path)
    image_times = image_times - image_times[0]  # Use time relative to first timestamp
    image_dt = np.mean(np.diff(image_times))

    # Convert coordinates to UTM (x,y)
    x,y = crs_tools.convert_long_lat_to_utm(lcf_long,lcf_lat)

    # Calculate along-track velocity vector, and corresponding unit vector
    vx_alongtrack = (x[-1] - x[0]) / (lcf_times[-1] - lcf_times[0])
    vy_alongtrack = (y[-1] - y[0]) / (lcf_times[-1] - lcf_times[0])
    v_alongtrack = np.array((vx_alongtrack,vy_alongtrack))
    u_alongtrack = v_alongtrack / np.linalg.norm(v_alongtrack)
    
    # Calculate cross-track unit vector
    u_crosstrack = np.array([-u_alongtrack[1],u_alongtrack[0]]) # Rotate 90 clockwise: (x,y) -> (-y,x)

    # Calculate length of pushbroom "footprint" on ground
    altitude = np.mean(lcf_alt) + altitude_offset
    L = pushbroom_width_on_ground(opening_angle_deg=camera_opening_angle_deg,
                                  relative_altitude=altitude)
    
    # Calculate "origin" (coordinates for upper left pixel in image)
    r_origin = np.array([x[0], y[0]]) - (L/2)*u_crosstrack
    C,F = r_origin

    # Calculate B (x-skew) and E (y-scale)
    B = vx_alongtrack*image_dt
    E = vy_alongtrack*image_dt

    # Calculate A (x-scale) and D (y-skew)
    cross_track_gsd = L/n_crosstrack_pixels  # Distance between pixels cross-track
    A,D = cross_track_gsd * u_crosstrack

    # Save to file
    world_file_parameters = np.array((A,D,B,E,C,F))
    print(f'{world_file_parameters=}')
    np.savetxt(world_file_path,world_file_parameters,fmt='%.12f')