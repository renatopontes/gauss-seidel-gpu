[x, text]=fscanfMat("..\out\matriz.txt")

[nl,nc]=size(x)

xset ("colormap",jetcolormap(256)) // use of the color table

grayplot(0:(nl-1), 0:(nc-1), x)

contour2d(0:(nl-1), 0:(nc-1), x, (nl+nc)/5)
