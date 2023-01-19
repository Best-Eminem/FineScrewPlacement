import copy
import numpy as np
import math
import os
import PySide2
# dirname = os.path.dirname(PySide2.__file__)
# plugin_path = os.path.join(dirname, 'plugins', 'platforms')
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
#
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction,PiecewiseFunction
import SimpleITK as sitk

# path = 'E:\\ZN-CT-nii\\data\\gt\\36.nii.gz' #path为文件的路径
# image = sitk.ReadImage(path)           #利用sampleItk读入mha数据，读入办法不唯一
# image = sitk.GetArrayFromImage(image)  #获得图像的数组形式的数据，需要注意顺序
# from preprocess.transform import resize_image_itk


def point2area_distance(Ax, By, Cz, D, point):
    """
    :param point1:数据框的行切片，三维
    :param point2:
    :param point3:
    :param point4:
    :return:点到面的距离
    """
    #Ax, By, Cz, D = define_area(point1, point2, point3,point4)
    mod_d = Ax * point[:,0] + By * point[:,1] + Cz * point[:,2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(mod_d) / mod_area
    return d
def define_area(points_plane):
    """
    法向量    :n={A,B,C}
    :return: (Ax, By, Cz, D)代表: Ax + By + Cz + D = 0
    """
    point1,point2,point3,point4 = points_plane
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    point4 = np.asarray(point4)
    AB = np.asmatrix(point2 - point1)
    CD = np.asmatrix(point4 - point3)
    N = np.cross(AB, CD)  # 向量叉乘，求法向量
    # Ax+By+Cz
    Ax = N[0, 0]
    By = N[0, 1]
    Cz = N[0, 2]
    D = -(Ax * point3[0] + By * point3[1] + Cz * point3[2])
    return Ax, By, Cz, D
def draw_model(image,sagittal_points,pedical_points):
    # x, y, z, value = np.random.randint(50,100,(4, 40))
    # print(x,y,z,value)

    # center_points = []
    # for i in range(5):
    #     a, b, c, d = points[i * 8 + 0], points[i * 8 + 1], points[i * 8 + 2], points[i * 8 + 3]
    #     coner_points = np.asarray([a, b, c, d])
    #     center = np.mean(coner_points, axis=0)
    #     center_points.append(center)
    # center_points = np.asarray(center_points, dtype='int32')

    #画特征点
    # for i in range(5):
    #     x = points[8 * i+0:8 * i+8, 0]
    #     y = points[8 * i+0:8 * i+8, 1]
    #     z = points[8 * i+0:8 * i+8, 2]
    #     point = mlab.points3d(x, y, z, scale_factor=5)
        #mlab.outline()
    #画矢状面中心点
    # x = center_points[:, 0]
    # y = center_points[:, 1]
    # z = center_points[:, 2]
    # point = mlab.points3d(x, y, z, scale_factor=5)



    # for points in [sagittal_points,pedical_points]:
    #     x = points[:,0]
    #     y = points[:,1]
    #     z = points[:,2]
    #     points = mlab.points3d(x, y, z, scale_factor=5)

    # spine = mlab.contour3d(image,color=(1,1,0),opacity=0.1,transparent=False) #draw model without background opacity:透明度值越小，越透明
    spine = mlab.contour3d(image, colormap='Oranges', opacity=0.4,
                           transparent=False)  # draw model without background opacity:透明度值越小，越透明

    # # draw model with the background
    # vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(image), name='3-d ultrasound ')
    # ctf = ColorTransferFunction()  # 该函数决定体绘制的颜色、灰度等
    # #for gray_v in range(256):
    #
    # ctf.add_rgb_point(-1, 0, 0, 0)
    # ctf.add_rgb_point(0.5, 1, 1, 1)
    # vol._volume_property.set_color(ctf)  # 进行更改，体绘制的colormap及color
    # vol._ctf = ctf
    # vol.update_ctf = True
    #
    # otf = PiecewiseFunction()
    # otf.add_point(0.4,0.01)
    # vol._otf = otf
    # vol._volume_property.set_scalar_opacity(otf)
    # # Also, it might be useful to change the range of the ctf::
    # ctf.range = [-1, 1]
    mlab.show()
def draw_plane(image,sagittal_points,pedical_points,spacing):
    position_smaller = []
    normal_vectors = []
    for i in range(5):
        plane_points = []
        img = image.copy()
        sagittal_tp = sagittal_points[i*4:i*4+4]
        sagittal_tp = sorted(sagittal_tp,key=lambda x: x[2]) #根据z轴从小到大排序
        pedical_tp = pedical_points[i*4:i*4+4]
        #plane_points.extend(points[6*i+2:6*i+6])
        plane_points.extend([sagittal_tp[-1],sagittal_tp[-2]])
        plane_points.append(np.mean(pedical_tp[0:2], axis=0))
        plane_points.append(np.mean(pedical_tp[2:], axis=0))
        Ax, By, Cz, D = define_area(plane_points)
        normal_vectors.append([Ax, By, Cz])

        position = np.argwhere(img == 1)
        d = point2area_distance(Ax, By, Cz, D, position) #计算mask中点到面的距离
        p_position_larger = np.argwhere(d>0.5)
        #p_position_smaller = np.argwhere(d<=0.5)
        position_larger = position[p_position_larger.reshape(p_position_larger.shape[0])]
        #position_smaller.append(position[p_position_smaller.reshape(p_position_smaller.shape[0])])
        img[position_larger[:,0],position_larger[:,1],position_larger[:,2]] = 0
        small = np.argwhere(img == 1)
        position_smaller.append(small)
        # pedical_plane = mlab.contour3d(img, color=(0, 0.5, 0.5)) # draw the plain for the pedicals
    position_smaller = np.asarray(position_smaller)
    #img[position_smaller[:, 0], position_smaller[:, 1], position_smaller[:, 2]] = 1
    draw_model(image, sagittal_points,pedical_points)
    # mlab.outline()
    # mlab.axes(xlabel='x', ylabel='y', zlabel='z')
    draw_nail(image, sagittal_points,pedical_points,normal_vectors,spacing)
    # draw_bounding_box(sagittal_points,pedical_points,normal_vectors,spacing) #依据矢状面预测点中上侧远离追弓根的那个点，框体长度120，高度40，进行倾斜并绘制
    mlab.show()

def draw_nail(image,sagittal_points,pedical_points,normal_vectors,spacing):
    #计算平面夹角时，由于用法向量进行计算，需要添加符号位，来表示倾斜的方向
    #以下两个参数分别为L5到L1的，第一个是钉子平行的旋转角度，第二个是钉子的头尾延伸多长
    center_rotate_angle = [[-15,-15],
                           [-5,-5],
                           [-5,-5],
                           [-5,-5],
                           [-5,-5]]
    nail_length = [[[30,50],[30,50]],
                   [[30,40],[30,40]],
                   [[30,50],[30,50]],
                   [[30,50],[30,50]],
                   [[30,50],[30,50]]]
    nail_radius = [[4,4],
                   [4,4],
                   [4,4],
                   [4,4],
                   [4,4]]
    mask_xyz = np.zeros((3,image.shape[0],image.shape[1],image.shape[2]))
    for i in range(mask_xyz[0].shape[0]):
        mask_xyz[0][i,:,:] = i
    for i in range(mask_xyz[1].shape[1]):
        mask_xyz[1][:,i,:] = i 
    for i in range(mask_xyz[2].shape[2]):
        mask_xyz[2][:,:,i] = i

    sign_bit = -1
    for i in range(5):
        if i >=2:
            sign_bit = 1
        nail_points = []
        sagittal_tp = sagittal_points[i * 4:i * 4 + 4]
        sagittal_tp = sorted(sagittal_tp, key=lambda x: x[2])  # 根据z轴从小到大排序
        pedical_tp = pedical_points[i * 4:i * 4 + 4]
        nail_points.append(np.mean(pedical_tp[0:2],axis=0,dtype=np.int32).tolist())
        nail_points.append(np.mean(pedical_tp[2:], axis=0,dtype=np.int32).tolist())
        nail_points = np.asarray(nail_points,dtype='int32')
        nail_point_left = nail_points[0] if nail_points[0,0]<nail_points[1,0] else nail_points[1]
        nail_point_right = nail_points[1] if nail_points[0,0]<nail_points[1,0] else nail_points[0]
        nail_length_head = nail_length[i][0][0]
        nail_length_tail = nail_length[i][0][1]
        left_points = np.zeros((nail_length_head+nail_length_tail,3))
        left_points[:, 0] = [nail_point_left[0]] * (nail_length_head+nail_length_tail)
        left_points[:, 1] = range(nail_point_left[1]-(nail_length_head), nail_point_left[1]+(nail_length_tail))
        left_points[:, 2] = [nail_point_left[2]] * (nail_length_head+nail_length_tail)
        #让钉子向下偏转
        left_points = structureRotateLineByAngle(left_points.tolist(),
                                                            [nail_point_left[0]-25, nail_point_left[1],
                                                             nail_point_left[2]],
                                                            [nail_point_left[0]+25, nail_point_left[1],
                                                             nail_point_left[2]],
                                                            sign_bit * getAngleFromVectors(np.asarray([0,0,1]),np.asarray(normal_vectors[i])))
        
        # 让钉子向横截面中心偏转
        # center_rotate_angle = -15
        left_points = np.asarray(structureRotateLineByAngle(left_points,
                                                            [nail_point_left[0], nail_point_left[1],
                                                             nail_point_left[2] - 25],
                                                            [nail_point_left[0], nail_point_left[1],
                                                             nail_point_left[2] + 25], center_rotate_angle[i][0]), dtype='int32')
        left_line_vector = nail_point_left - left_points[0]

        dist_mat = spine2line(mask_xyz, image, nail_point_left, left_line_vector,spacing)
        dist_radius = getDistRadius(mask_xyz,nail_point_left,spacing)
        dist_radius[np.where(image == 1)] = np.inf
        max_radius_left = np.min(dist_radius) if np.min(dist_radius)>0 else 3
        dist_ = image + np.where(dist_mat > max_radius_left, 0, 2)
        online_points = np.where(dist_ == 3)
        start_point, end_point = getStartEndPoints(online_points)
        left_line_vector = start_point - end_point
        left_points = []
        y_low = end_point[1] if start_point[1] > end_point[1] else start_point[1]
        y_high = end_point[1] if start_point[1] < end_point[1] else start_point[1]

        extent_length = 30 if i!=0 else 20
        head_extent = 10
        for y in range(y_low+head_extent, y_high+extent_length):
            x = (y - start_point[1])/left_line_vector[1] * left_line_vector[0] + start_point[0]
            z = (y - start_point[1])/left_line_vector[1] * left_line_vector[2] + start_point[2]
            left_points.append([x,y,z])
        left_points = np.asarray(left_points,dtype=np.int32)

        # 画钉子的起点和顶点
        # points = mlab.points3d(start_point[0], start_point[1], start_point[2], scale_factor=5)
        # points = mlab.points3d(end_point[0], end_point[1], end_point[2], scale_factor=5)

        # 计算当前直线允许的最大半径、直线长度
        line_len = getDistanceBetweenTwoPoints(mask_xyz[:, start_point[0], start_point[1], start_point[2]],
                              mask_xyz[:, end_point[0], end_point[1], end_point[2]],spacing)
        print(max_radius_left * 2, line_len)
        
        #left_points = structureRotateLineByAngle(left_points, [nail_point_left[0] - 10],[0, 0, nail_point_left[0] + 10], 0, 0, 15)


        nail_length_head = nail_length[i][1][0]
        nail_length_tail = nail_length[i][1][1]
        right_points = np.zeros((nail_length_head+nail_length_tail, 3))
        right_points[:, 0] = [nail_point_right[0]] * (nail_length_head+nail_length_tail)
        right_points[:, 1] = range(nail_point_right[1] - (nail_length_head), nail_point_right[1] + (nail_length_tail))
        right_points[:, 2] = [nail_point_right[2]] * (nail_length_head+nail_length_tail)
        right_points = structureRotateLineByAngle(right_points.tolist(),
                                                             [nail_point_right[0]- 25, nail_point_right[1],
                                                              nail_point_right[2]],
                                                             [nail_point_right[0] + 25, nail_point_right[1],
                                                              nail_point_right[2]], sign_bit * getAngleFromVectors(np.asarray([0,0,1]),np.asarray(normal_vectors[i])))
        right_points = np.asarray(structureRotateLineByAngle(right_points,
                                                             [nail_point_right[0], nail_point_right[1],
                                                              nail_point_right[2] - 25],
                                                             [nail_point_right[0], nail_point_right[1],
                                                              nail_point_right[2] + 25], 0-center_rotate_angle[i][1]), dtype='int32')
        # left_points = structureRotateLineByAngle(left_points, [nail_point_left[0] - 10],[0, 0, nail_point_left[0] + 10], 0, 0, 15)
        right_line_vector = nail_point_right - right_points[0]

        dist_mat = spine2line(mask_xyz, image, nail_point_right, right_line_vector,spacing)
        dist_radius = getDistRadius(mask_xyz,nail_point_right,spacing)
        dist_radius[np.where(image == 1)] = np.inf
        max_radius_right = np.min(dist_radius) if np.min(dist_radius) > 0 else 3
        dist_ = image + np.where(dist_mat > max_radius_right, 0, 2)
        online_points = np.where(dist_ == 3)
        start_point, end_point = getStartEndPoints(online_points)
        right_line_vector = start_point - end_point
        right_points = []
        y_low = end_point[1] if start_point[1]>end_point[1] else start_point[1]
        y_high = end_point[1] if start_point[1]<end_point[1] else start_point[1]

        for y in range(y_low+head_extent, y_high+extent_length):
            x = (y - start_point[1]) / right_line_vector[1] * right_line_vector[0] + start_point[0]
            z = (y - start_point[1]) / right_line_vector[1] * right_line_vector[2] + start_point[2]
            right_points.append([x, y, z])
        right_points = np.asarray(right_points,dtype=np.int32)
        # 画钉子的起点和顶点
        # points = mlab.points3d(start_point[0], start_point[1], start_point[2], scale_factor=5)
        # points = mlab.points3d(end_point[0], end_point[1], end_point[2], scale_factor=5)

        # 计算当前直线允许的最大半径、直线长度
        line_len = getDistanceBetweenTwoPoints(mask_xyz[:, start_point[0], start_point[1], start_point[2]],
                              mask_xyz[:, end_point[0], end_point[1], end_point[2]],spacing)
        print(max_radius_right * 2, line_len)
        max_radius = min(max_radius_left,max_radius_right)
        line_left = mlab.plot3d(left_points[:, 0], left_points[:, 1], left_points[:, 2], tube_radius=max_radius*0.8 if i!=0 else max_radius ,
                                tube_sides=20, color=(1, 0, 0))  # colormap='Spectral')
        line_right = mlab.plot3d(right_points[:, 0], right_points[:, 1], right_points[:, 2], tube_radius=max_radius*0.8 if i!=0 else max_radius, tube_sides=20,color=(1,0,0))

    #mlab.show()
def getDistanceBetweenPoints(p1, p2,spacing):
    return np.sqrt((p1[:,0] - p2[:,0])*spacing[0] ** 2 + (p1[:,1] - p2[:,1]) *spacing[1] ** 2 + (p1[:,2] - p2[:,2])*spacing[2] ** 2)
def getDistanceBetweenTwoPoints(p1, p2,spacing):
    return math.sqrt(((p1[0] - p2[0])*spacing[0])  ** 2 + ((p1[1] - p2[1]) *spacing[1]) ** 2 + ((p1[2] - p2[2]) *spacing[2]) ** 2)
# 空间中，一个结构式围绕一条直线line旋转angle角之后。求旋转之后的结构式上所有点的坐标
# 已知直线上两个点为p1，p2。两个点确定一条直线。无所谓结构式上的点是否在此直线上。
# 参考：https://www.cnblogs.com/leejxyz/p/5250935.html
# 参考网页中错误较多，需要改正
# 我不知道具体的原理是什么，反正结果是正确的。

def structureRotateLineByAngle(listPoint, p1, p2, angle):
    distance = getDistanceBetweenTwoPoints(p1, p2,spacing=[1,1,1])
    # 方向向量(u,v,w)需为单位向量！！！请仔细查看什么是单位向量，怎么计算单位向量中每个方向的值
    # 网页中的单位向量用的是sqrt(3) /3。这是不对的
    # 注意，如果p1和p2这两个点重叠，则无法计算单位向量
    u = (p1[0] - p2[0]) / distance
    v = (p1[1] - p2[1]) / distance
    w = (p1[2] - p2[2]) / distance

    SinA = math.sin(angle * math.pi / 180)
    CosA = math.cos(angle * math.pi / 180)

    uu = u * u
    vv = v * v
    ww = w * w
    uv = u * v
    uw = u * w
    vw = v * w

    t00 = uu + (vv + ww) * CosA
    t10 = uv * (1 - CosA) + w * SinA
    t20 = uw * (1 - CosA) - v * SinA

    t01 = uv * (1 - CosA) - w * SinA
    t11 = vv + (uu + ww) * CosA
    t21 = vw * (1 - CosA) + u * SinA

    t02 = uw * (1 - CosA) + v * SinA
    t12 = vw * (1 - CosA) - u * SinA  # 网页中的t12被写成t21
    t22 = ww + (uu + vv) * CosA

    a0 = p2[0]
    b0 = p2[1]
    c0 = p2[2]

    t03 = (a0 * (vv + ww) - u * (b0 * v + c0 * w)) * (1 - CosA) + (b0 * w - c0 * v) * SinA
    t13 = (b0 * (uu + ww) - v * (a0 * u + c0 * w)) * (1 - CosA) + (c0 * u - a0 * w) * SinA
    t23 = (c0 * (uu + vv) - w * (a0 * u + b0 * v)) * (1 - CosA) + (a0 * v - b0 * u) * SinA

    # 新的坐标的list
    newlistPoint = []

    for p in listPoint:
        newPoint = [t00 * p[0] + t01 * p[1] + t02 * p[2] + t03,
                         t10 * p[0] + t11 * p[1] + t12 * p[2] + t13,
                         t20 * p[0] + t21 * p[1] + t22 * p[2] + t23]

        newlistPoint.append(newPoint)

    # 返回旋转之后的坐标list
    return newlistPoint
def getAngleFromVectors(normal1,normal2):
    # 引入numpy模块并创建两个向量
    # 分别计算两个向量的模：
    # l_x = np.sqrt(vector_a.dot(vector_a))
    # l_y = np.sqrt(vector_b.dot(vector_b))
    # #print('向量的模=', l_x, l_y)
    #
    # # 计算两个向量的点积
    # dot = vector_a.dot(vector_b)
    # #print('向量的点积=', dian)
    #
    # # 计算夹角的cos值：
    # cos_ = dot / (l_x * l_y)
    # #print('夹角的cos值=', cos_)
    #
    # # 求得夹角（弧度制）：
    # radian = np.arccos(cos_)
    # #print('夹角（弧度制）=', radian)
    data_M = np.sqrt(np.sum(normal1 * normal1))
    data_N = np.sqrt(np.sum(normal2 * normal2))
    cos_theta = np.sum(normal1 * normal2) / (data_M * data_N)
    theta = np.degrees(np.arccos(cos_theta))  # 角点b的夹角值
    if theta > 90:
        theta = 180 - theta
    if theta > 15:
        theta = 15
    # print('theta: ',theta)
    return theta
def plot3Dboxes(corners):
    for i in range(corners.shape[0]):
        corner = corners[i]
        plot3Dbox(corner)

def plot3Dbox(corner):
    idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
    x = corner[0, idx]
    y = corner[1, idx]
    z = corner[2, idx]
    mlab.plot3d(x, y, z, color=(0.23, 0.6, 1), colormap='Spectral', representation='wireframe', line_width=5)
    mlab.show()
def get_points_from_point(point,distence):
    a = point.copy()
    a[0] = a[0] - distence
    b = point.copy()
    b[0] = b[0] + distence
    return a,b
def draw_bounding_box(sagittal_points,pedical_points,normal_vectors,spacing):
    # 依据矢状面预测点中上侧远离追弓根的那个点，框体长度120，高度40，宽度50，进行倾斜并绘制
    box_length = 120
    box_height = 35
    box_width = 80
    sign_bit = -1
    bounding_points_pair = []
    for i in range(5):
        if i >=2:
            sign_bit = 1
        #point_a, point_b = points[6 * i + 0:6 * i + 2]
        sagittal_tp = sagittal_points[i * 4:i * 4 + 4]
        sagittal_tp = sorted(sagittal_tp, key=lambda x: x[2])  # 根据z轴从小到大排序
        point_a, point_b = sagittal_tp[-2:]
        if point_a[1] > point_b[1]:
            temp = point_b.copy()
            point_b = point_a
            point_a = temp
        point_a[1] -= 2;point_a[2]+=10
        #point_a[2] += (2 if point_a[2] >= point_b[2] else 5)
        point_b = point_a.copy();point_b[1] += box_length;
        point_c = point_a.copy();point_c[2] -= box_height
        point_d = point_b.copy();point_d[2] -= box_height

        # if point_c[1] < point_d[1]:
        #     point_d = get_point_from_line(point_c, point_d)
        # else:
        #     point_c = get_point_from_line(point_c, point_d)

        point_a_left, point_a_right = get_points_from_point(point_a, box_width)
        point_b_left, point_b_right = get_points_from_point(point_b, box_width)
        point_c_left, point_c_right = get_points_from_point(point_c, box_width)
        point_d_left, point_d_right = get_points_from_point(point_d, box_width)

        box_center_point = np.mean((point_a,point_b,point_c,point_d),axis=0)
        box_center_point_left,box_center_point_right = get_points_from_point(box_center_point, box_width)
        #让八个顶点旋转
        vertex_points = [point_a_left, point_a_right,
                        point_b_left, point_b_right,
                        point_c_left, point_c_right,
                        point_d_left, point_d_right]

        point_a_left, point_a_right,point_b_left, point_b_right,point_c_left, point_c_right,point_d_left, point_d_right = structureRotateLineByAngle(vertex_points,
                                                 point_a_left,
                                                 point_a_right,
                                                 sign_bit * getAngleFromVectors(np.asarray([0, 0, 1]),
                                                                                np.asarray(normal_vectors[i])))

        #12条棱
        bounding_points_pair.append([point_a_left,point_a_right])
        bounding_points_pair.append([point_b_left, point_b_right])
        bounding_points_pair.append([point_c_left, point_c_right])
        bounding_points_pair.append([point_d_left, point_d_right])

        bounding_points_pair.append([point_a_left, point_b_left])
        bounding_points_pair.append([point_a_right, point_b_right])
        bounding_points_pair.append([point_c_left, point_d_left])
        bounding_points_pair.append([point_c_right, point_d_right])

        bounding_points_pair.append([point_a_left, point_c_left])
        bounding_points_pair.append([point_a_right, point_c_right])
        bounding_points_pair.append([point_b_left, point_d_left])
        bounding_points_pair.append([point_b_right, point_d_right])

    bounding_points_pair = np.asarray(bounding_points_pair,dtype='int32')
    for i in range(bounding_points_pair.shape[0]):
        points_pair  = bounding_points_pair[i]
        mlab.plot3d(points_pair[:, 0], points_pair[:, 1], points_pair[:, 2], color=(1, 1, 1), tube_radius=2,colormap='Spectral')
        # points = mlab.points3d(points_pair[:, 0], points_pair[:, 1], points_pair[:, 2], scale_factor=5)
    # mlab.show()
def get_point_from_line(point_a,point_b):
    #给定一个y坐标，求空间直线上一个点的坐标
    vector = point_a - point_b
    a,b,c = vector
    if point_a[1] < point_b[1]:
        k = 100 / b
        x = k * a + point_a[0]
        z = k * c + point_a[2]
        y = point_a[1] + 100
    else:
        k = 100 / b
        x = k * a + point_b[0]
        z = k * c + point_b[2]
        y = point_b[1] + 100
    return [x,y,z]
def getDistRadius(mask_xyz, pedical_point,spacing):
    xLine_mat = (mask_xyz[0, :, :, :] - pedical_point[0])*spacing[2]
    yLine_mat = (mask_xyz[1, :, :, :] - pedical_point[1])*spacing[2]
    zLine_mat = (mask_xyz[2, :, :, :] - pedical_point[2])*spacing[2]

    radius_mat = np.power(xLine_mat, 2) + np.power(yLine_mat, 2) + np.power(zLine_mat, 2)
    return np.sqrt(radius_mat)
def spine2line(spine_xyz,spine, cpoint, dirpoint,spacing):
    '''
    计算脊柱数据上各点与直线的距离
    Input:
        - spine_xyz: [3,M,N,P] world coordinate
        - spine: [M,N,P] mask
        - cpoint: [x,y,z], crossing fixed point
        - dirpoint: [x,y,z], oritation vector
    Return:
        - dist_mat: [M,N,P], distance matrix
    '''
    # indexs = np.indices(spine.shape)
    xLine_mat = np.ones(spine.shape) * cpoint[0]
    yLine_mat = np.ones(spine.shape) * cpoint[1]
    zLine_mat = np.ones(spine.shape) * cpoint[2]
    xOrita_mat = np.ones(spine.shape) * dirpoint[0]
    yOrita_mat = np.ones(spine.shape) * dirpoint[1]
    zOrita_mat = np.ones(spine.shape) * dirpoint[2]
    xLine_mat = (spine_xyz[0, :, :, :] - xLine_mat)*spacing[0]
    yLine_mat = (spine_xyz[1, :, :, :] - yLine_mat)*spacing[1]
    zLine_mat = (spine_xyz[2, :, :, :] - zLine_mat)*spacing[2]

    dist_mat = np.power((yLine_mat * zOrita_mat - yOrita_mat * zLine_mat), 2) + np.power(
        (zLine_mat * xOrita_mat - zOrita_mat * xLine_mat), 2) + np.power(
        (xLine_mat * yOrita_mat - yLine_mat * xOrita_mat), 2)
    dist_mat = np.sqrt(dist_mat)
    dirpoint_sqrt = np.sqrt(np.sum(np.power(dirpoint,2)))
    return dist_mat/dirpoint_sqrt
def distEuclid(point1, point2,spacing):
    # 计算两点距离
    return np.linalg.norm(point2-point1)
def getStartEndPoints(online_points):
    minxv, minxv_ind = np.min(online_points[0]), np.where(online_points[0] == np.min(online_points[0]))
    maxxv, maxxv_ind = np.max(online_points[0]), np.where(online_points[0] == np.max(online_points[0]))
    minyvs = online_points[1][minxv_ind]
    minzvs = online_points[2][minxv_ind]
    maxyvs = online_points[1][maxxv_ind]
    maxzvs = online_points[2][maxxv_ind]
    if np.min(minyvs)<=np.max(maxyvs):
        minyv = np.min(minyvs)
        maxyv = np.max(maxyvs)
    else:
        minyv = np.max(minyvs)
        maxyv = np.min(maxyvs)

    if np.min(minzvs)<=np.max(maxzvs):
        minzv = np.min(minzvs)
        maxzv = np.max(maxzvs)
    else:
        minzv = np.max(minzvs)
        maxzv = np.min(maxzvs)
    # start_point = np.array([maxxv,maxyv,maxzv])
    # end_point = np.array([minxv,minyv,minzv])
    start_point = np.array([minxv, minyv, minzv])
    end_point = np.array([maxxv, maxyv, maxzv])
    return start_point, end_point
def lineInVerterbra(spine_xyz,spine, dist, radiu_thres=None):
    # 计算当前直线允许的最大半径、直线长度
    # line_len = distEuclid(spine_xyz[:,start_point[0],start_point[1],start_point[2]],
    #                       spine_xyz[:,end_point[0],end_point[1],end_point[2]])

    dist_ = copy.deepcopy(dist)
    dist_[np.where(spine==1)] = np.inf
    if radiu_thres is not None:
        # Todo 如果后面需要排除尾部的
        dist_[0:radiu_thres[0], :, :] = np.inf
        dist_[radiu_thres[1] + 1:, :, :] = np.inf
    # dist_ = dist_[start_point[0]:end_point[0]+1,:,:]
    max_radius = np.min(dist_)
    # 以下用于可视化验证
    # ppp = np.where(dist_ == max_radius)
    return max_radius



if __name__ == '__main__':
    
    dirs= ['/home/gpu/Spinal-Nailing/FineScrewPlacement/spinningup/spinningup_imgs_test/test_pre.nii.gz',
        '/home/gpu/Spinal-Nailing/FineScrewPlacement/spinningup/spinningup_imgs_test/sub-verse835_dir-iso_L1_seg-vert_msk.nii.gz',
           '/home/gpu/Spinal-Nailing/FineScrewPlacement/spinningup/spinningup_imgs_test/sub-verse835_dir-iso_L2_seg-vert_msk.nii.gz',
           '/home/gpu/Spinal-Nailing/FineScrewPlacement/spinningup/spinningup_imgs_test/sub-verse835_dir-iso_L3_seg-vert_msk.nii.gz',
           '/home/gpu/Spinal-Nailing/FineScrewPlacement/spinningup/spinningup_imgs_test/sub-verse835_dir-iso_L4_seg-vert_msk.nii.gz',
           '/home/gpu/Spinal-Nailing/FineScrewPlacement/spinningup/spinningup_imgs_test/sub-verse821_L5_seg-vert_msk.nii.gz']
    # dirs.reverse
    whole_sections = sitk.ReadImage(dirs[0])
    whole_sections = sitk.GetArrayFromImage(whole_sections)
    whole_sections = np.transpose(whole_sections, (2, 1, 0))
    # for dir in dirs[1:]:
    #     image_pre_sections = sitk.ReadImage(dir)
    #     # image_pre_sections = resize_image_itk(image_pre_sections, origin_image.GetSize(), resamplemethod=sitk.sitkLinear)
    #     image_pre_sections = sitk.GetArrayFromImage(image_pre_sections)
    #     image_pre_sections = np.transpose(image_pre_sections, (2, 1, 0))
    #     whole_sections = np.append(image_pre_sections, whole_sections, axis=2)


    whole_sections[whole_sections == 2] = 3
    # image_pre_sections[image_pre_sections !=1] = 0


    # pedical_points = np.asarray(pedical_points)
    # temp = pedical_points[:, 0].copy()
    # pedical_points[:, 0] = pedical_points[:, 2]
    # pedical_points[:, 2] = temp
    # pedical_points[:,2] = pedical_points[:,2]+3
    # for i in range(len(pedical_points)):
    #     if i % 4 <2:
    #         pedical_points[i,0] = pedical_points[i,0]+3

    draw_model(whole_sections,sagittal_points=None,pedical_points=None)

    #plot3Dboxes(corners)
    #draw_bounding_box(points)

    print(1)


