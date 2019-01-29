# find the nearest point by kd-tree

## algroithim

> Understand Recursive Better

1. First, we define three points: target point, current nearest point(CNP), doubt point.
2. Second, we initial three point : target point -> target point, the leaf node -> current nearest point, father node -> doubt point.
3. Third, we compare the distance between target point and current point (d1) with that between target point and doubt point (d2).
	(a) if d1 > d2, we will assign the doubt point to CNP, what's more, we will check if CNP has a brother, if true,give CNP's brother node to doubt point,if not,give CNP's grandfather to doubt point
	(b) if d1 < d2 and doubt point has a father node, we just give the father node of doubt point to the doubt point 
	(c) if d1 < d2 and doubt point dont have a father node, we get the Nearest point