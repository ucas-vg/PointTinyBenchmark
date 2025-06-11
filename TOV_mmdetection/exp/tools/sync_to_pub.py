import os
os.system("copy README.md ..\\pub\\TOV_mmdetection")
os.chdir("..\\pub\\TOV_mmdetection")

os.system("git add -A && git commit -m \"update readme\" && git push")
