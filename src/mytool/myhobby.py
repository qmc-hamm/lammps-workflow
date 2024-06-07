"""
笔记
----------
我的爱好哈哈哈
"""
import math

def cal_bmi(weight,height):
    """
    功能
    ----------
    计算BMI

    参数
    ----------
    weight:体重(单位:斤)
    height:身高(单位:cm)

    返回值
    ----------
    BMI

    例子
    ----------
    cal_bmi(130,168)
    返回值:23.0
    """
    return weight/2/(height/100)**2

def cal_lap_speed(speed):
    """
    功能
    ----------
    由配速计算圈速

    参数
    ----------
    speed:配速(3位数字)

    返回值
    ----------
    lap_speed:圈速(3位数字)

    例子
    ----------
    cal_lap_speed("435")
    返回值:"150"
    """
    minute=int(str(speed)[0])
    second=int(str(speed)[1:])
    lap_speed=(minute*60+second)/2.5/60
    if math.ceil(60*(lap_speed-int(lap_speed)))==60:
        return "%.1d00"%(math.ceil(lap_speed))
    else:
        return "%.1d%.2d"%(int(lap_speed),math.ceil(60*(lap_speed-int(lap_speed))))

def cal_speed(running,time):
    """
    功能
    ----------
    计算配速

    参数
    ----------
    running:跑的距离(单位:米)
    time:时间("hhmmss")

    返回值
    ----------
    speed:配速(每公里配速,"mss")

    例子
    ----------
    cal_speed(10000,"005500")
    返回值:"530"
    """
    running=running/1000.0
    hour=int(time[:2])
    minute=int(time[2:4])
    second=int(time[4:])
    total_second=hour*3600+minute*60+second
    speed=total_second/60.0/running
    if math.ceil(60*(speed-int(speed)))==60:
        return "%.1d00"%(math.ceil(speed))
    else:
        return "%.1d%.2d"%(int(speed),math.ceil(60*(speed-int(speed))))

def to_second(time):
    """
    功能
    ----------
    计算秒数

    参数
    ----------
    time:时间("hhmmss")

    返回值
    ----------
    秒数(s)

    例子
    ----------
    to_second("002000")
    返回值:1200
    """
    hour=int(time[:2])
    minute=int(time[2:4])
    second=int(time[4:])
    return hour*3600+minute*60+second

def to_time(total_second):
    """
    功能
    ----------
    计算时间("hhmmss")

    参数
    ----------
    total_second:秒数(s)

    返回值
    ----------
    时间("hhmmss")

    例子
    ----------
    to_time(1200)
    返回值:"002000"
    """
    hour=(total_second-total_second%3600)/3600
    minute=((total_second-hour*3600)-(total_second-hour*3600)%60)/60
    second=total_second-hour*3600-minute*60
    return "%.2d%.2d%.2d"%(hour,minute,second)
