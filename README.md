# Ercot-Grid-Data-Visualization
Python script to visualize frequency oscillation points in the Texas ERCOT power grid.

The script itself relies on a number of input files that can be found in the inputs folder.

## What this program tries to accomplish
  There are censors scattered across the Texas ERCOT power grid called 'buses'. These buses collect power grid frequency oscillation data.
  Each bus collects thousands of data points per day (roughly 4000), and it's important to visualize this data to get an understanding
  of the health of the grid. 

  To visualize this data, the Python script reads oscillation data for a given day and turns it into a video with a graph and a heat map
  of the state of Texas. Here is one of the frames:
  
<img width="500" height="400" src="Screen Shot 2023-08-22 at 12.37.48 AM.png">

  Each line on the graph represents a bus and each triangle on the map is also a corresponding bus.
  On the right, you can also see different colorations on the map itself. If you look at the slide bar key on the right of the map, 
  you can see that different colors correspond to different levels of oscillation intensity. So in the image, it's blue toward the bottom 
  bus because it's experiencing intense oscillations.
  Using interpolation, I'm able to correspond the color shading to specific areas on the map that are experiencing oscillations (the blue 
  fades in the image).
  The sliding bar on the right of the map and the red dot on the graph represents the mean frequency for that given point in time.

  You can go to this link to view the whole video: https://drive.google.com/drive/folders/1YD1qlsZ6W5iJnMOGyFfVl5tJQoP9ilgQ?usp=drive_link

## The Code Explained
### Left Half (Oscillation Graph)
Using pandas, I read in the oscillation_data.xlsx file and used matplotlib to represent each column of the data file as a line on a graph. The first column of the oscillation file contains all the time of day (military time) that is used to make the x axis/x values. To create the red line that the dot tracks on, I used the "find_median()" function to get the average oscillation value for each given
time of day and plotted those values with matplotlib. The median array is then iterated over, a dot is plotted at every point, each plot containing a dot is converted to an openCV image object using the function "figure2cv2img2(fig)", and finally each image object is concatenated to get the video effect.

### Right Half (Color Map)
The outline of the Ercot grid(isn't the exact outline of Texas) and the triangles (bus locations) come from the data files ERCOT.txt and BusLocations.xlsx respectively. Each file is read with pandas and plotted via matplotlib.  
For each median oscillation value, the "build_map()" function is called and returns an openCV image object to be concatenated into video. In the build_map() function:

1. Oscillation values are extracted for each bus
2. Bus and oscillation data are filtered based on if they fall inside the grid outline
3. Nearest neighbor interpolation is used with the scipy library and a grid file (clip_area_ERCOT.txt) is read in to interpolate on
4. A colored contour plot is generated based on the interpolated data (matplotlib and numpy)





