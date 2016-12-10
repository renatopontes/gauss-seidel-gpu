[x, text]=fscanfMat("C:\Users\Renato\Google Drive\ufrj\7_periodo\programacao paralela com gpu\codigo\trabalho_GPU\out.txt")

[nl,nc]=size(x)

xset ("colormap",jetcolormap(256)) // use of the color table

grayplot(0:(nl-1), 0:(nc-1), x)

contour2d(0:(nl-1), 0:(nc-1), x, (nl+nc)/5)
