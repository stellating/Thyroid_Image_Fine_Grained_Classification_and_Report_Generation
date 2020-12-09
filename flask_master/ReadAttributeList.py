def ReadAttributeList(AttributeFile):
    AttributeList={}
    AttributeNameList=[]
    flag=0
    with open(AttributeFile,'r') as f:
        lines = f.readlines()
        for c,line in enumerate(lines):
            if line[0]!=' ':
                AttributeList[line.strip()]=[]
                AttributeNameList.append(line.strip())
                flag=c
            else:
                AttributeList[lines[flag].strip()].append(line.strip())
    return AttributeNameList,AttributeList

if __name__ == '__main__':
    AttributeFile='new_edition_utf8.txt'
    list,dict=ReadAttributeList(AttributeFile)
    print(list,dict)
