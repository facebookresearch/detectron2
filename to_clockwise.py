#-*-coding:utf-8 -*-
import json


def getFlag(plist = []):
    area = 0
    if(len(plist)<3):
        return 0
    p0 = plist[0]
    for i in range(1,len(plist) - 1):
        p1,p2 = plist[i],plist[i+1]
        v1x = p1[0] - p0[0]
        v1y = p1[1] - p0[1]
        v2x = p2[0] - p0[0]
        v2y = p2[1] - p0[1]
        s = 0.5 * (v1x * v2y - v1y * v2x)
        area += s
    return -1 if area >0 else 1


def getData(fname = 'tmp.json'):
    with open(fname,'r',encoding='utf-8') as f:
#         str = f.read()
#         return json.loads(str)
        info=json.loads(f.read())
        #对于坐标点的数目大于20个点的，让其坐标为20个点
        for key in info.keys():
            each=info[key]
            for i in range(len(each)):
                if len(info[key][i]['points'])>50:
                    split=int(len(info[key][i]['points'])/50)
                    info[key][i]['points']=info[key][i]['points'][::split]
        info['res_3102']={}
        return info

def main():
    results = {}
    dataDict = getData()
    for k,v in dataDict.items():
       # k = 'res_925'
        #v = dataDict[k]
        r = []
        for i,data in enumerate(v):
            plist= data['points']
            flag = getFlag(plist)
            if(flag)>0:
                data['points'] = list(reversed(plist))
           # print(plist)
           # print(flag)
            #print(data['points'])
            r.append(data)
        #return
        results[k] = r
    with open('results.json','w',encoding= 'utf-8') as f:
        f.write(json.dumps(results))

    return

main()
