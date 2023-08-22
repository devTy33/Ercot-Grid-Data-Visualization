# Ercot-Grid-Data-Visualization
Python script to visualize frequency oscillation points in the Texas ERCOT power grid.

The script itself relies on a number of input files that can be found in the src folder.

<h2> What this program tries to accomplish </h2>
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
