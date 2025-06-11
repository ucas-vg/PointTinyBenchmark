from mmdet.models.dense_heads.cpr_head import CirclePtFeatGenerator
import matplotlib.pyplot as plt


gen = CirclePtFeatGenerator(5, num_classes=80)
depend, pts = gen.get_depend(4, return_pts=True)

plt.scatter(pts[:, 0].tolist(), pts[:, 1].tolist())
for i, dep in enumerate(depend.tolist()):
    for j in dep:
        pt1, pt2 = pts[i], pts[j]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.show()
