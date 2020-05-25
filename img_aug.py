from PIL import Image,ImageChops
import matplotlib.pyplot as plt
import cv2
#origin_img=Image.open(r'D:\pycharm_community\python_workspace\image_transformation/1.jpg')
origin_img=cv2.imread(r'D:\pycharm_community\python_workspace\image_transformation/1.jpg')
print(origin_img.shape)
# # zoom_rang
# origin_size=origin_img.size
# img = origin_img.resize((2*origin_size[0]//3,origin_size[1]//3),resample=Image.LANCZOS)
# plt.title("zoom_range")
# plt.imshow(img)
# plt.show()

#shear_range
# img = origin_img.crop((400, 700, 1500, 1500))  # 参数为坐标左上右下
# plt.imshow(img)
# plt.title("shear_range")
# plt.show()

#width_shift/height_shift
# img = ImageChops.offset(origin_img, 100, 50)
# width, height =img.size
# img.paste((0, 0, 0), (0, 0, 100, height))
# img.paste((0, 0, 0), (0, 0, width, 50))
# plt.title("shift")
# plt.imshow(img)
# plt.show()
