import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import namedtuple

def coorSpace2LngLat(ppoint, R=None):
    # 空间坐标变为经纬度
    # input: ppoint(x,y,z)
    # Output: Lat, Lng
    if isinstance(ppoint, np.ndarray):
        ppoint = np.asarray(ppoint)
    if R is None:
        R = np.linalg.norm(ppoint)
    return np.arcsin(ppoint[2] / R), np.arctan(ppoint[1] / ppoint[0])

def coorLngLat2Space(angles_L, angles_R=None, R=1., default = False):
    """返回旋转后的方向向量

    Args:
        angles (_type_): (Lat, Lng), rad
        R (_type_, optional): _description_. Defaults to 1..
        default (bool, optional): _description_. Defaults to True.

    Returns:
        Output: ppoint(x,y,z)
    """
    x_L = R * np.cos(np.deg2rad(angles_L[1])) * np.sin(np.deg2rad(angles_L[0]))
    y_L = R * np.cos(np.deg2rad(angles_L[1])) * np.cos(np.deg2rad(angles_L[0]))
    z_L = R * np.sin(np.deg2rad(angles_L[1]))
    if default:
        if x_L>=0:
            x_L = -x_L
            y_L = -y_L
    if angles_R != None:
        x_R = R * np.cos(np.deg2rad(angles_R[1])) * np.sin(np.deg2rad(angles_R[0]))
        y_R = R * np.cos(np.deg2rad(angles_R[1])) * np.cos(np.deg2rad(angles_R[0]))
        z_R = R * np.sin(np.deg2rad(angles_R[1]))
        if default:
            if x_R>=0:
                x_R = -x_R
                y_R = -y_R
        return np.array([x_L, y_L, z_L]), np.array([x_R, y_R, z_R])
    else: return np.array([x_L, y_L, z_L])

def distEuclid(point1, point2):
    # 计算两点距离
    return np.linalg.norm(point2-point1)

def spine2line(spine_xyz, spine, cpoint, dirpoint):
    '''
    计算脊柱数据上各点与直线的距离
    
    Input:
        - spine_xyz: [3,M,N,P] world coordinate
        - spine: [M,N,P] mask
        - cpoint: [x,y,z], crossing fixed point
        - dirpoint: [x,y,z], orientation vector
        
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
    xLine_mat = spine_xyz[0, :, :, :] - xLine_mat
    yLine_mat = spine_xyz[1, :, :, :] - yLine_mat
    zLine_mat = spine_xyz[2, :, :, :] - zLine_mat

    dist_mat = np.power((yLine_mat * zOrita_mat - yOrita_mat * zLine_mat), 2) + np.power(
        (zLine_mat * xOrita_mat - zOrita_mat * xLine_mat), 2) + np.power(
        (xLine_mat * yOrita_mat - yLine_mat * xOrita_mat), 2)
    dist_mat = np.sqrt(dist_mat)
    return dist_mat

def getEndpoint(spine_xyz, spine, cpoint, dirvector, R=0.8, line_thres=None, point_dist=None, line_dist=None):
    '''
    计算直线两端点坐标
    Input:
        - spine:
        - cpoint:
        - dirvector: direction vector
        - R:
        - threshold:[ymin,ymax]
        - dist:
    '''
    dist_ = spine + np.where(line_dist>R, 0, 2)
    if line_thres is not None:
        # Todo 如果后面需要排除尾部
        dist_[0:line_thres[0], :, :] = 0
        dist_[line_thres[1]+1:, :,:] = 0

    online_points = list(np.where(dist_==3))
    uncontinuous_bound = None
    for index in range(len(online_points[0])-1):
        if (online_points[0][index+1] - online_points[0][index]) > 10:
            uncontinuous_bound = index
            break
    if uncontinuous_bound!=None:
        online_points[0] = online_points[0][:index+1]
        online_points[1] = online_points[1][:index+1]
        online_points[2] = online_points[2][:index+1]
    # if len(online_points[0]) == 0:
    #     print('cpoint:%d %d %d'% (cpoint[0], cpoint[1],cpoint[2]))
    x_axis_dis = np.max(online_points[0]) - np.min(online_points[0])
    y_axis_dis = np.max(online_points[1]) - np.min(online_points[1])
    # x_axis_dis = np.max(online_points[0]) - np.min(online_points[0])
    if x_axis_dis > y_axis_dis:
        minxv, minxv_ind = np.min(online_points[0]), np.where(online_points[0] == np.min(online_points[0]))
        maxxv, maxxv_ind = np.max(online_points[0]), np.where(online_points[0] == np.max(online_points[0]))
        minyvs = online_points[1][minxv_ind]
        minzvs = online_points[2][minxv_ind]
        maxyvs = online_points[1][maxxv_ind]
        maxzvs = online_points[2][maxxv_ind]
        if np.min(minyvs)<np.max(maxyvs):
            minyv = np.min(minyvs)
            maxyv = np.max(maxyvs)
        else:
            minyv = np.max(minyvs)
            maxyv = np.min(maxyvs)

        if np.min(minzvs)<np.max(maxzvs):
            minzv = np.min(minzvs)
            maxzv = np.max(maxzvs)
        else:
            minzv = np.max(minzvs)
            maxzv = np.min(maxzvs)
    else:
        minyv, minyv_ind = np.min(online_points[1]), np.where(online_points[1] == np.min(online_points[1]))
        maxyv, maxyv_ind = np.max(online_points[1]), np.where(online_points[1] == np.max(online_points[1]))
        minxvs = online_points[0][minyv_ind]
        minzvs = online_points[2][minyv_ind]
        maxxvs = online_points[0][maxyv_ind]
        maxzvs = online_points[2][maxyv_ind]
        if np.min(minxvs)<np.max(maxxvs):
            minxv = np.min(minxvs)
            maxxv = np.max(maxxvs)
        else:
            minxv = np.max(minxvs)
            maxxv = np.min(maxxvs)
        if np.min(minzvs)<np.max(maxzvs):
            minzv = np.min(minzvs)
            maxzv = np.max(maxzvs)
        else:
            minzv = np.max(minzvs)
            maxzv = np.min(maxzvs)
    start_point = np.array([minxv, minyv, minzv])
    end_point = np.array([maxxv, maxyv, maxzv])
    if distEuclid(start_point, end_point) <= 10:
        print('error')
    return start_point, end_point


def lineInVerterbra(spine_xyz,spine, cpoint, start_point, end_point, point_dist, line_dist, radiu_thres=None):
    """_summary_

    Args:
        spine_xyz (numpy): spine coards.
        spine (numpy): mask array.
        start_point (numpy): start point of screw.
        end_point (numpy): end point of screw.
        dist (numpy): distences matrix among screw line and every points of mask array.
        radiu_thres (_type_, optional): radius threshold. Defaults to None.

    Returns:
        max_radius: max radius of screw.
        line_len: the length of screw.
        (ppp[0],ppp[1],ppp[2]): screw points in max_radius.
    """
    # 计算当前直线允许的最大半径、直线长度
    line_len = distEuclid(spine_xyz[:,start_point[0],start_point[1],start_point[2]],
                          spine_xyz[:,end_point[0],end_point[1],end_point[2]])

    # dist_ = copy.deepcopy(point_dist)
    dist_ = copy.deepcopy(line_dist)  #计算在line上的点时，应该用spine相对于line的距离？
    dist_[np.where(spine==1)] = np.inf
    if start_point[1] >= end_point[1]:
        max_y, min_y = start_point[1], end_point[1]
    else: 
        max_y, min_y = end_point[1], start_point[1]
    # 计算spine相对于line的距离，只取line上的部分点到spine的距离来计算螺钉最大半径
    dist_[:, 0:int((min_y+cpoint[1])/2), :] = np.inf
    dist_[:, int((max_y+cpoint[1])/2)+1:, :] = np.inf
    
    if radiu_thres is not None:
        # Todo 如果后面需要排除尾部的
        dist_[0:radiu_thres[0], :, :] = np.inf
        dist_[radiu_thres[1] + 1:, :, :] = np.inf
    # dist_ = dist_[start_point[0]:end_point[0]+1,:,:]
    max_radius = np.min(dist_) if np.inf > np.min(dist_) > 1.0 else 1.0
    # 以下用于可视化验证
    line_dist[np.where(spine<1)] = np.inf #此行代码可以保证螺钉在椎体内
    ppp = np.where(line_dist < max_radius)
    x_ppp, y_ppp, z_ppp = [], [], []
    if start_point[0] >= end_point[0]:
        max_x, min_x = start_point[0], end_point[0]
    else: 
        max_x, min_x = end_point[0], start_point[0]
    for index in range(len(ppp[0])):
        if max_x >= ppp[0][index] >= min_x:
            x_ppp.append(ppp[0][index])
            y_ppp.append(ppp[1][index])
            z_ppp.append(ppp[2][index])
    return max_radius, line_len, (np.array(x_ppp),np.array(y_ppp),np.array(z_ppp))

def getLenRadiu(spine_xyz, spine, cpoint, dirvector, R=0.8, line_thres=None, radiu_thres = None, point_dist=None, line_dist=None):
    # 计算
    # radiu_thres is used to truncate x axis of the start/end point
    if point_dist is None: point_dist = spine2point(spine_xyz, spine, cpoint)
    if line_dist is None: line_dist = spine2line(spine_xyz, spine, cpoint, dirvector)
    start_point, end_point = getEndpoint(spine_xyz, spine, cpoint, dirvector, R=R, line_thres= line_thres, point_dist=point_dist, line_dist=line_dist)
    max_radius, line_len, radiu_p = lineInVerterbra(spine_xyz, spine, cpoint, start_point, end_point, point_dist = point_dist, line_dist = line_dist, radiu_thres=radiu_thres)
    return max_radius, line_len,{'start_point':start_point,'end_point':end_point,'radiu_p':radiu_p, 'cpoint':cpoint}

def pointInSphere(chk_point, sphere):
    # check whether the point in the sphere
    ppdist = np.sqrt(np.sum(np.power(chk_point - sphere[0],2)))
    return ppdist <= sphere[1]

def irc2xyz_arr(coord_irc, origin_xyz, vxSize_xyz,  direction_a):
    # 体素坐标转为空间坐标 (带旋转)
    origin_a = np.tile(np.expand_dims(origin_xyz, [1, 2, 3]), coord_irc.shape[1:])
    vxSize_a = np.expand_dims(np.array(vxSize_xyz),[1,2,3])
    direction_a = np.array(direction_a).reshape(3, 3)
    direction_a_0 = np.tile(np.expand_dims(direction_a[0,:], [1, 2, 3]), coord_irc.shape[1:])
    direction_a_1 = np.tile(np.expand_dims(direction_a[1,:], [1, 2, 3]), coord_irc.shape[1:])
    direction_a_2 = np.tile(np.expand_dims(direction_a[2,:], [1, 2, 3]), coord_irc.shape[1:])
    coord_irc = coord_irc * vxSize_a
    coords_xyz = np.concatenate([np.sum(direction_a_0 * coord_irc, axis=0, keepdims=True),
                                np.sum(direction_a_1 * coord_irc, axis=0, keepdims=True),
                                np.sum(direction_a_2 * coord_irc, axis=0, keepdims=True)],0)
    coords_xyz = coords_xyz + origin_a
    return coords_xyz

def irc2real_arr(spine_shape, spacing, origin_irc=None):
    # 体素坐标转为世界坐标 (不带旋转)
    # spine_shape: shape of spine array / spine array
    # spacing: spacing of the spine
    # origin_irc: IRC of the supposed origin in spine array. if None, do not translate.
    if not isinstance(spine_shape,(list,tuple)):
        spine_shape = spine_shape.shape
    coords = np.indices(spine_shape)
    spacing = np.expand_dims(np.array(spacing), [1, 2, 3])
    coords = coords * spacing
    if origin_irc is None:
        return coords
    else:
        return ChangeOrigin(coords, origin_irc)

def ChangeOrigin(coords_xyz, center_irc):
    # 将center_irc设为原点
    # Change the origin of coordinates
    # Input: - coords_xyz: format:[3,m,n,t], ndarray
    #        - center_irc: [3,], ndarray
    center_xyz = coords_xyz[:, center_irc[0], center_irc[1], center_irc[2]]
    center_xyz = np.tile(np.expand_dims(center_xyz, [1, 2, 3]), coords_xyz.shape[1:])
    return coords_xyz - center_xyz

def spine2point(spine_xyz, spine, point):
    '''
    计算脊柱数据上各点与定点的距离
    Input:
        - spine_xyz: [3,M,N,P] world coordinate
        - spine: [M,N,P] mask
        - point: [x,y,z], fixed point
    Return:
        - dist_mat: [M,N,P], distance matrix
    '''
    x_mat = np.ones(spine.shape) * point[0]
    y_mat = np.ones(spine.shape) * point[1]
    z_mat = np.ones(spine.shape) * point[2]
    x_mat = spine_xyz[0, :, :, :] - x_mat
    y_mat = spine_xyz[1, :, :, :] - y_mat
    z_mat = spine_xyz[2, :, :, :] - z_mat
    dist_mat = np.sqrt(np.power(x_mat, 2) + np.power(y_mat, 2) + np.power(z_mat, 2))
    dist_mat[np.where(spine==1)] = np.inf
    return dist_mat