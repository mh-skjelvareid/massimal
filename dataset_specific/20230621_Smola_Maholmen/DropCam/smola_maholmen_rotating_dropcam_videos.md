# A note on rotating videos

Drop-camera videos for transects 1 and 2 originally appeared "upside down", probably because the GoPro was set to automatic rotation. 

The easiest way to set the correct rotation for the vidoes seemed to be using exiftool. ffmpeg can also be used, but it seems that most alternatives involve transcoding the video. exiftool just changes the metadata, and also has a very simple syntax.

To set the correct rotation for all videos in a folder, I cd'ed into the folder and used

    exiftool -rotation=0 *.MP4

Not quite sure why "rotation=0" works here (one would expect "rotation=180") - perhaps the rotation is defined relative to the original camera frame of reference? Anyway, it works in this case.

After rotating all videos, I found that video 5 from transect 2 was actually in the correct orientation originally.