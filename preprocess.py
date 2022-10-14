import numpy as np

def GaussianHighlight(image, coordinates, diameter):
  new_image = np.zeros(image.shape, dtype=np.uint64)
  if diameter%2 == 0:
    diameter = diameter +1
  arr = np.zeros((diameter,diameter,3), dtype=np.uint64)
  imgsize = arr.shape[:2]
  innerColor = (255,255, 255)
  outerColor = (0, 0, 0)


  # kernel = gkern(l=diameter)
  # kernel = kernel * 255

  for y in range(imgsize[1]):
    for x in range(imgsize[0]):
        #Find the distance to the center
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)

        #Make it on a scale from 0 to 1innerColor
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)

        #Calculate r, g, and b values
        # r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)
        # g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)
        # b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)

        r = 255
        g = 255
        b = 255
        
        # r = kernel[y,x]
        # g = kernel[y,x]
        # b = kernel[y,x]

        arr[y, x] = (int(r), int(g), int(b))
  for coordinate in coordinates:
    upper_first = int(coordinate[0]- diameter//2) 
    upper_first_offset = 0
    if upper_first < 0:
      upper_first_offset = abs(upper_first)
      upper_first = 0
      
    lower_first = int(coordinate[0] + diameter//2)
    lower_first_offset = diameter - 1
    if lower_first > new_image.shape[0]:
      lower_first_offset = lower_first_offset - (lower_first - new_image.shape[0])
      lower_first = new_image.shape[0]

    left_second = int(coordinate[1] - diameter//2)
    left_second_offset = 0
    if left_second < 0:
      left_second_offset = abs(left_second)
      left_second = 0

    right_second = int(coordinate[1] + diameter//2)
    right_second_offset = diameter - 1
    if right_second > new_image.shape[1]:
      right_second_offset = right_second_offset - (right_second - new_image.shape[1])
      right_second = new_image.shape[1]

    new_image[upper_first:lower_first, left_second:right_second] += arr[upper_first_offset:lower_first_offset,left_second_offset:right_second_offset,0]
  new_image = new_image.clip(0,255)
  
  return new_image