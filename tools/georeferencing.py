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



def read_lcf_file(lcf_file_path, time_rel_to_file_start = True):
    """ Read location files (.lcf) generated by Resonon Airborne Hyperspectral imager 
    
    # Input arguments:
    lcf_file_path:
        Path to lcf file. Usually a "sidecar" file to an hyperspectral image 
        with same "base" filename.

    # Returns:
    lcf_data:
        Dictionary with keys describing the type of data, and data
        formatted as numpy arrays. All arrays have equal length.
        
        The 7 types of data:
        - 'time': System time in seconds, relative to some (unknown)
        starting point. Similar to "Unix time" (seconds since January 1. 1970),
        but values indicate starting point around 1980. The values are 
        usually offset to make the first timestamp equal to zero.
        See flag time_rel_to_file_start.
        - 'roll': Roll angle in radians, positive for "right wing up"
        - 'pitch': Pitch angle in radians, positive nose up  
        - 'yaw': (heading) in radians, zero at due North, PI/2 at due East
        - 'longitude': Longitude in decimal degrees, negative for west longitude
        - 'latitude': Latitude in decimal degrees, negative for southern hemisphere
        - 'altitude': Altitude in meters relative to the WGS-84 ellipsiod.
        
    time_rel_to_file_start:
        Boolean indicating if first timestamp should be subtracted from each
        timestamp, making time relative to start of file.

    # Notes:
    - The LCF file format was shared by Casey Smith at Resonon on February 16. 2021.
    - The LCF specification was inherited from Space Computer Corp.
    """

    # Load LCF data
    lcf_raw = np.loadtxt(lcf_file_path)
    column_headers = ['time','roll','pitch','yaw','longitude','latitude','altitude']
    lcf_data = {header:lcf_raw[:,i] for i,header in enumerate(column_headers)}
    
    if time_rel_to_file_start:
        lcf_data['time'] -= lcf_data['time'][0]

    return lcf_data


def read_times_file(times_file_path,time_rel_to_file_start=True):
    """ Read image line timestamps (.times) file generated by Resonon camera 
    
    # Input arguments:
    times_file_path:
        Path to times file. Usually a "sidecar" file to an hyperspectral image 
        with same "base" filename.
    time_rel_to_file_start:
        Boolean indicating if times should be offset so that first
        timestamp is zero. If not, the original timestamp value is returned.

    # Returns:
    times:
        Numpy array containing timestamps for every line of the corresponding 
        hyperspectral image. The timestamps are in units of seconds, and are 
        relative to when the system started (values are usually within the 
        0-10000 second range). If time_rel_to_file_start=True, the times
        are offset so that the first timestamp is zero.
        
        The first timestamp of the times file and the  first timestamp of the 
        corresponding lcf file (GPS/IMU data) are assumed to the 
        recorded at exactly the same time. If both sets of timestamps are 
        offset so that time is measured relative to the start of the file,
        the times can be used to calculate interpolated GPS/IMU values
        for each image line.

    """
    image_times = np.loadtxt(times_file_path)
    if time_rel_to_file_start:
        image_times = image_times - image_times[0]
    return image_times


def calculate_pushbroom_imager_transform(time,longitude,latitude,altitude,framerate, 
                                         ground_altitude=0, pitch_offset=0,roll_offset=0,
                                         n_crosstrack_pixels=900, camera_opening_angle_deg=36.5,
                                         use_world_file_ordering=False):
    """ Calculate simple affine transform to map pushbroom raster image to UTM coordinates 
    
    # Input parameters:
    time:
        Time in seconds as measured by GPS/IMU (array)
    longitude:
        Longitude in decimal degrees (WGS-84) (array)
    latitude:
        Latitude in decimal degrees (WGS-84) (array)
    altitude:
        Altitude in meters above WGS-84 ellipsoid (array)
    framerate:
        Pushbroom camera framerate (lines per second)

    # Keyword parameters:
    ground_altitude:
        Altitude of ground in meters, measured in same coordinate system
        as altitude.  Relative height of camera relative to ground is calculated as 
        mean(altitude) - ground_altitude. 
        Note that the altitude as measured by the GPS may be offset from the true
        altitude above sea level, both because of measurement bias and because
        the WGS ellipsiod does not match the sea level perfectly. The ground altitude
        can be used as a "tuning" parameters to correct for this.
        If an image appears to be too small on the ground, decrease ground_altitude.
        In the image appears too big, increase ground_altitude.
    pitch_offset:
        Pitch is assumed to be close to zero (camera looking straight down).
        Set pitch_offset (radians) to compensate for small deviations.
        Camera looking slightly forward corresponds to positive pitch_offset. 
    roll_offset:
        Roll is assumed to be close to zero (camera looking straight down).
        Set roll_offset (radians) to compensate for small deviations.
        Camera looking slightly to the right corresponds to positive roll_offset. 
    n_crosstrack_pixels:
        Number of pixels "cross-track" (also called number of "samples")
    camera_opening_angle_deg:
        Opening angle of pushbroom camera in degrees.
    use_world_file_ordering:
        Boolean, indicating that transform parameters are returned in same
        order as "world files" (A,D,B,E,C,F). 
    
    # Returns:
    transform:
        6-element tuple with "world file" affine transformation
        from raster image indices (xi = column index, yi = row index) to 
        UTM coordinates (x = easting, y = northing):
            x = A*ix + B*iy + C
            y = E*ix + E*iy + F
        If use_world_file_ordering = True, parameters are returned in the order
        (A,D,B,E,C,F) used in ESRI "world files". If not, they are returned
        in alphabetical order, (A,B,C,D,E,F).
    utm_epsg:
        EPSG code for UTM zone used in transform.

    # Notes:
    - The UTM zone is inferred from the latitude / longitude coordinates.  
    - When world file ordering is used, a world file can be saved simply
    using numpy.savetxt:
        transform_parameters = calculate_pushbroom_imager_transform(...,
                                use_world_file_ordering=True)
        numpy.savetxt(path_to_world_file,transform_parameters)
    """

    # Convert coordinates to UTM (x,y)
    x,y,utm_epsg = crs_tools.convert_long_lat_to_utm(longitude,latitude,return_utm_epsg=True)

    # Calculate along-track velocity vector, and corresponding unit vector
    vx_alongtrack = (x[-1] - x[0]) / (time[-1] - time[0])
    vy_alongtrack = (y[-1] - y[0]) / (time[-1] - time[0])
    v_alongtrack = np.array((vx_alongtrack,vy_alongtrack))
    u_alongtrack = v_alongtrack / np.linalg.norm(v_alongtrack)
    
    # Calculate cross-track unit vector
    u_crosstrack = np.array([-u_alongtrack[1],u_alongtrack[0]]) # Rotate 90 clockwise: (x,y) -> (-y,x)

    # Calculate length of pushbroom "footprint" on ground
    relative_altitude = np.mean(altitude) - ground_altitude
    L = pushbroom_width_on_ground(
        opening_angle_deg=camera_opening_angle_deg,
        relative_altitude=relative_altitude)
    
    # Calculate "origin" (coordinates for upper left pixel in image)
    r_origin = np.array([x[0], y[0]])  # Start in camera position
    r_origin -= (L/2)*u_crosstrack     # Offset to edge, corresponds to image origin 
    r_origin += relative_altitude*math.tan(pitch_offset) * u_alongtrack  # Correct for pitch offset
    r_origin -= relative_altitude*math.tan(roll_offset) * u_crosstrack   # Correct for roll offset
    C,F = r_origin

    # Calculate B (x-skew) and E (y-scale)
    image_dt = 1/framerate    # Image line sampling period
    B = vx_alongtrack*image_dt
    E = vy_alongtrack*image_dt

    # Calculate A (x-scale) and D (y-skew)
    cross_track_gsd = L/n_crosstrack_pixels  # Distance between pixels cross-track
    A,D = cross_track_gsd * u_crosstrack

    # Return
    if use_world_file_ordering:
        return (A,D,B,E,C,F), utm_epsg
    else:
        return (A,B,C,D,E,F), utm_epsg
    

def world_file_from_lcf_times_files(lcf_file_path,times_file_path,world_file_path,**kwargs):
    """ Calculate world file (affine transformation) for raster image based on .lcf and .times files 
    
    # Arguments:
    lcf_file_path:
        Path to .lcf file, containing camera IMU and GNSS information
    times_file_path:
        Path to .times file. 
    world_file_path:
        Path for output world file. Should have file extension corresponding to
        image format, e.g. 'pgw' for PNG files, or, alternatively,
        just 'wld' (accepted by GDAL and QGIS). Uses UTM coordinates,
        with local UTM zone estimated from coordinates in lcf file. 
         
    # Keyword arguments:
    Keyword parameters are passed on to calculate_pushbroom_imager_transform().
    Note: pitch_offset and roll_offset are estimated as mean values from LCF files,
    and can (currently) not be set manually.  Other keyword arguments to the 
    function (e.g. ground_altitude) can be passed in. 

    """

    lcf_data = read_lcf_file(lcf_file_path)
    image_times = read_times_file(times_file_path)
    framerate = 1/np.mean(np.diff(image_times))

    # Calculate transform parameters
    affine_transform_parameters, _ = calculate_pushbroom_imager_transform(
        lcf_data['time'],
        lcf_data['longitude'],
        lcf_data['latitude'],
        lcf_data['altitude'],
        framerate,
        use_world_file_ordering=True,
        pitch_offset = np.mean(lcf_data['pitch']),
        roll_offset = np.mean(lcf_data['roll']),
        **kwargs)

    # Save to file
    np.savetxt(world_file_path,affine_transform_parameters)