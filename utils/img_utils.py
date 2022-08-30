import cv2
import os
import glob
import imageio
# glob.glob(r"E:/Picture/*/*.jpg")

def images_to_video(path, suffix, isDelete=False, savename =None):
    img_array = []
    imgList = glob.glob(os.path.join(path,suffix))
    imgList.sort(key = lambda x: int(x.split('\\')[-1].split('.')[0].split('_')[-1])) # sorted by name
    for filename in imgList:
        img = cv2.imread(filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)

    if savename is None:
        savename = imgList[0].split('\\')[-1].split('.')[0].split('_')[0]
    imageio.mimsave(os.path.join(path,savename+'.gif'), img_array, 'GIF', fps=2)
    # 删除原图，
    if isDelete:
        for filename in imgList:
            os.remove(filename)
        print("Delete Images")

if __name__ == "__main__":
    # path = r"C:\Users\Lenovo\Desktop\testimg"  # 改成你自己图片文件夹的路径
    # suffix = '*.jpg'
    # images_to_video(path, suffix)
    images_to_video('.\\imgs', '*.jpg', True)

