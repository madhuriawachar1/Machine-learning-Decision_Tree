N = Number of samples
M = Number of attributes
d = depth of the tree
c = max number of classes, only in case of discreate input

Case 1: Real Input Real Output
Theoretical Fit Time Complexity: O(MN^2) for creating each tree node, so total time taken is O(2^d*(MN^2))
 please refer figure_3 AND figure_4  
Theoretical Predict Time Complexity: O(d) for creating each tree node

Case 2: Real Input Discreate Output
Theoretical Fit Time Complexity: O(MN^2) for creating each tree node, so total time taken is O(2^d*(MN^2))
 please refer figure_5 AND figure_6
Theoretical Predict Time Complexity: O(d) for creating each tree node

Case 3: Discreate Input Discreate Output
Theoretical Fit Time Complexity: O(MN) for creating each tree node, so total time taken is O(c^d*(MN))
 please refer figure_7 AND figure_8
Theoretical Predict Time Complexity: O(d) for creating each tree node

Case 4: Discreate Input Real Output
Theoretical Fit Time Complexity: O(MN) for creating each tree node, so total time taken is O(c^d*(MN))
 please refer figure_9 AND figure_10
Theoretical Predict Time Complexity: O(d) for creating each tree node