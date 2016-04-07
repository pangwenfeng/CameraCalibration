#encode:utf-8
'''
Created on 2016��3��24��

@author: Administrator
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

class CameraCalibration:
    
    #载入数据
    #file_name_array 存有相应文件路径的数组，当前的顺序为0：观测数据（2D) 1：模型数据（3D)
    def loadData(self,file_name_array):
        data=[]
        line=[]
        lines=[]
        for file_name in file_name_array:
            with open(file_name,'r') as fr:
                tmp=[inst.strip().split('   ') for inst in fr.readlines()]#对数据的排列结构较为严格，这里可以改进为自动识别提取数据
                for row in tmp:
                    for col in row:
                        line.append(int(col))
                    lines.append(line)
                    line=[]
            data.append(lines)
            lines=[]
        return  data
    
    #计算转换矩阵
    #extra_col 是否需要认为在数据的最后一列添加一列1
    def getProMatrix(self,observe_data,model_data,extra_col=True):
        observe_data=np.matrix(observe_data)
        model_data=np.matrix(model_data)
        observe_data=np.hstack((observe_data,np.ones((np.size(observe_data,0),1))))
        model_data=np.hstack((model_data,np.ones((np.size(model_data,0),1))))
        #print(model_data)
        P=np.linalg.lstsq(model_data, observe_data)
        print(P[0])
        return P[0].T
    
    #计算摄像机模型的参数，包括4个内参，7个外参，1个尺度参数
    def getCameraParameters(self,P):
        a=P[:,0:3]
        b=P[:,-1]
        rho=1/np.linalg.norm(a[2])
        r3=rho*a[2]
        #print(r3)
        #print(a[2,:])
        u0=rho**2*(np.dot(a[0],a[2].T))
        v0=rho**2*(np.dot(a[1],a[2].T))
        cos_theta=-np.dot(np.cross(a[0],a[2]),np.cross(a[1],a[2]).T)/\
                            (np.linalg.norm(np.cross(a[0],a[2]))*np.linalg.norm(np.cross(a[1],a[2])))
        sin_theta=np.sqrt(1-cos_theta**2)
        alpha=(rho**2)*np.linalg.norm(np.cross(a[0],a[2]))
        beta=(rho**2)*np.linalg.norm(np.cross(a[1],a[2]))*sin_theta
        r1=np.cross(a[1],a[2])/(np.linalg.norm(np.cross(a[1],a[2])))
        r2=np.cross(r3,r1)
        K=np.matrix([[alpha,-alpha*cos_theta/sin_theta,u0],[0,beta/sin_theta,v0],[0,0,1]])
        t=rho*np.linalg.inv(K)*b

    #验证所得的转换矩阵是否有效
    def valifyP(self,P):
        a=list(range(11))
        coord1=[]
        coord2=[]
        coord3=[]
        for c1 in a:
            for c2 in a:
                coord1.append([c1,c2,0,1])
                coord2.append([c1,10,c2,1])
                coord3.append([10,c1,c2,1])
        coord1=np.array(coord1)
        coord2=np.array(coord2)
        coord3=np.array(coord3)
        
        r1=(P*coord1.T).T
        #print(coord1)
        r2=(P*coord2.T).T        
        r3=(P*coord3.T).T        
        
        #使用3个集合进行验证
#        plt.figure(1)
        #print(r1)
#         plt.scatter(r1[:,0], r1[:,1], s=20, c='r', marker='o',lw=0)
#         plt.scatter(r2[:,0], r2[:,1], s=20, c='b', marker='o',lw=0)
#         plt.scatter(r3[:,0], r3[:,1], s=20, c='g', marker='o',lw=0)
#         plt.show()
        
        #使用一个立方体进行验证
#         p1=[0,1,1,1]
#         p2=[1,1,1,1]
#         p3=[0,0,1,1]
#         p4=[1,0,1,1]
#         p5=[0,1,0,1]
#         p6=[1,1,0,1]
#         p7=[0,0,0,1]
#         p8=[1,0,0,1]

#         p1=[0,1,1,1]
#         p2=[1,1,1,1]
#         p3=[0,1,0,1]
#         p4=[1,1,0,1]
#         p5=[0,0,1,1]
#         p6=[1,0,1,1]
#         p7=[0,0,0,1]
#         p8=[1,0,0,1]
#         
#         p=np.array([p1,p2,p3,p4,p5,p6,p7,p8])
#         cube1=(P*p.T).T
#         plt.figure(2)
#         plt.scatter(cube1[0:4,0], cube1[0:4,1], s=20, c='r', marker='o',lw=0)
#         plt.scatter(cube1[4:8,0], cube1[4:8,1], s=20, c='g', marker='o',lw=0)
#         plt.plot([cube1[0,0],cube1[1,0]],[cube1[0,1],cube1[1,1]],c='r')#1-2#这里的坐标应改为与数据集中的坐标一致，否则出现坐标混乱的现象
#         plt.plot([cube1[0,0],cube1[2,0]],[cube1[0,1],cube1[2,1]],c='r')#1-3
#         plt.plot([cube1[1,0],cube1[3,0]],[cube1[1,1],cube1[3,1]],c='r')#2-4
#         plt.plot([cube1[1,0],cube1[5,0]],[cube1[1,1],cube1[5,1]],c='r')#2-6
#         plt.plot([cube1[2,0],cube1[3,0]],[cube1[2,1],cube1[3,1]],c='r')#3-4
#         plt.plot([cube1[2,0],cube1[6,0]],[cube1[2,1],cube1[6,1]],c='r')#3-7
#         plt.plot([cube1[3,0],cube1[7,0]],[cube1[3,1],cube1[7,1]],c='r')#4-8
#         plt.plot([cube1[6,0],cube1[7,0]],[cube1[6,1],cube1[7,1]],c='r')#7-8
#         plt.plot([cube1[5,0],cube1[7,0]],[cube1[5,1],cube1[7,1]],c='r')#6-8
#        plt.show()
        
        img=np.ones((800,800,3),np.uint8)*255
        
        for tmp in r1:
            #print(tmp[0,0])
            cv2.circle(img,(np.int(tmp[0,0]),np.int(tmp[0,1])),4,(0,0,255),-1);
        for tmp in r2:
            #print(tmp[0,0])
            cv2.circle(img,(np.int(tmp[0,0]),np.int(tmp[0,1])),4,(255,0,0),-1);
        for tmp in r3:
            #print(tmp[0,0])
            cv2.circle(img,(np.int(tmp[0,0]),np.int(tmp[0,1])),4,(0,255,0),-1);

        cv2.imshow("Image",img)
        cv2.waitKey()
        
    def showCube(self,cube1,video_out):
#         plt.clf()
#         plt.scatter(cube1[0:4,0], cube1[0:4,1], s=20, c='r', marker='o',lw=0)
#         plt.scatter(cube1[4:8,0], cube1[4:8,1], s=20, c='g', marker='o',lw=0)
#         plt.plot([cube1[0,0],cube1[1,0]],[cube1[0,1],cube1[1,1]],c='r')#1-2#这里的坐标应改为与数据集中的坐标一致，否则出现坐标混乱的现象
#         plt.plot([cube1[0,0],cube1[2,0]],[cube1[0,1],cube1[2,1]],c='r')#1-3
#         plt.plot([cube1[1,0],cube1[3,0]],[cube1[1,1],cube1[3,1]],c='r')#2-4
#         plt.plot([cube1[1,0],cube1[5,0]],[cube1[1,1],cube1[5,1]],c='r')#2-6
#         plt.plot([cube1[2,0],cube1[3,0]],[cube1[2,1],cube1[3,1]],c='r')#3-4
#         plt.plot([cube1[2,0],cube1[6,0]],[cube1[2,1],cube1[6,1]],c='r')#3-7
#         plt.plot([cube1[3,0],cube1[7,0]],[cube1[3,1],cube1[7,1]],c='r')#4-8
#         plt.plot([cube1[6,0],cube1[7,0]],[cube1[6,1],cube1[7,1]],c='r')#7-8
#         plt.plot([cube1[5,0],cube1[7,0]],[cube1[5,1],cube1[7,1]],c='r')#6-8
#         
#         plt.ylim(0,1000)
#         plt.xlim(0,1000)
#         plt.show()

        #plt.scatter(cube1[0:4,0], cube1[0:4,1], s=20, c='r', marker='o',lw=0)
        #plt.scatter(cube1[4:8,0], cube1[4:8,1], s=20, c='g', marker='o',lw=0)
        
        img=np.ones((800,800,3),np.uint8)*255
        cv2.line(img,(np.int(cube1[0,0]),np.int(cube1[0,1])),(np.int(cube1[1,0]),np.int(cube1[1,1])),(255,0,0))#1-2#这里的坐标应改为与数据集中的坐标一致，否则出现坐标混乱的现象
        cv2.line(img,(np.int(cube1[0,0]),np.int(cube1[0,1])),(np.int(cube1[2,0]),np.int(cube1[2,1])),(255,0,0))#1-3
        cv2.line(img,(np.int(cube1[0,0]),np.int(cube1[0,1])),(np.int(cube1[4,0]),np.int(cube1[4,1])),(255,0,0))#1-5
        cv2.line(img,(np.int(cube1[1,0]),np.int(cube1[1,1])),(np.int(cube1[5,0]),np.int(cube1[5,1])),(255,0,0))#2-6
        cv2.line(img,(np.int(cube1[4,0]),np.int(cube1[4,1])),(np.int(cube1[6,0]),np.int(cube1[6,1])),(255,0,0))#5-7
        cv2.line(img,(np.int(cube1[2,0]),np.int(cube1[2,1])),(np.int(cube1[6,0]),np.int(cube1[6,1])),(255,0,0))#3-7
        cv2.line(img,(np.int(cube1[4,0]),np.int(cube1[4,1])),(np.int(cube1[5,0]),np.int(cube1[5,1])),(255,0,0))#5-6
        cv2.line(img,(np.int(cube1[6,0]),np.int(cube1[6,1])),(np.int(cube1[7,0]),np.int(cube1[7,1])),(255,0,0))#7-8
        cv2.line(img,(np.int(cube1[5,0]),np.int(cube1[5,1])),(np.int(cube1[7,0]),np.int(cube1[7,1])),(255,0,0))#6-8
        
        video_out.write(img)
        cv2.imshow("Image",img)  
        cv2.waitKey(100)
        
    def move(self,P,orignal_position,movement):
        #current_position=(P*orignal_position.T).T    
        #plt.figure()
        
        fourcc=cv2.VideoWriter_fourcc(*'XVID')
        out=cv2.VideoWriter('out_put.avi',fourcc,2,(800,800))
       
        for m in movement:
            #current_position+=m
            current_position=(P*orignal_position.T).T 
            self.showCube(current_position,out)
            #time.sleep(1)
            #print(m)
            #print(orignal_position)
            orignal_position+=m
            
        out.release()
        cv2.destroyAllWindows()
        
if __name__=='__main__':
    cameraC=CameraCalibration()
    data=cameraC.loadData(['observe.dat','model.dat'])
    #print(data)
    #print(data[0][1])
    P=cameraC.getProMatrix(data[0], data[1], True)
    #cameraC.getCameraParameters(P)
    cameraC.valifyP(P)
    
    #测试移动立方体
    p1=[0,1,1,1]
    p2=[1,1,1,1]
    p3=[0,1,0,1]
    p4=[1,1,0,1]
    p5=[0,0,1,1]
    p6=[1,0,1,1]
    p7=[0,0,0,1]
    p8=[1,0,0,1]       
    p=np.array([p1,p2,p3,p4,p5,p6,p7,p8])#原始位置
    movement=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
                       [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]])
    cameraC.move(P,p, movement) 
    
    