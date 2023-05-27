# #Lists 

# new_list = ['one', 'two', 'three', 'four', 'five', 'six']
# print(new_list)

# new_list.append('seven')
# print(new_list)

# print(new_list.pop())


# popped_item = new_list.pop()
# print(popped_item)
# print(new_list)

# new_list[0] = 'ONE ALL CAPS'
# print(new_list)

# new_list.pop(-1)
# print(new_list)




#sort
new_list = ['a', 'e', 'x', 'b', 'c']
num_list = [4, 1, 8, 3]

#new_list.sort()
#print(new_list)

#reverse and sort
# print(num_list)

# num_list.sort()
# print(num_list)

# num_list.reverse()
# print(num_list)




#dictionaries
# my_dict = {'key1':'value1','key2':'value2'}
# print(my_dict)
# print(my_dict['key1'])

# prices_lookup = {'apple':2.99,'oranges':1.99,'milk':5.80}
# print(prices_lookup['apple'])

# d = {'k1':123,'k2':[0,1,2,4],'k3':{'insideKey':100}}
# print(d['k2'])
# print(d['k2'][3])
# print(d['k3']['insideKey'])

# d = {'key1':['a','b','c']}
# print(d)
# mylist = d['key1']
# print(mylist)
# leter = mylist[2]
# print(leter)
# print(leter.upper())
# print(d['key1'][2].upper())

# d = {'k1':100, 'k2':200}
# print(d)
# d['k3']=300
# print(d)
# d['k1'] = 'NEW VALUE'
# print(d)
# print(d.keys())
# print(d.values())
# print(d.items())



#tuples
# t = (1, 2, 3)
# mylist = [1, 2, 3]
# print(type(t))
# print(type(mylist))
# print(len(t))

# t = ('one', 2)
# print(t[0])

# t = ('a', 'a', 'b','c')
# print(t.count('a'))
# print(t.index('c'))

# print(mylist)
# mylist[0] = 'new'
# print(mylist[0])
# t[0] = 'NEW'



#sets
# myset = set()
# print(myset)
# myset.add(1)
# print(myset)
# myset.add(2)
# print(myset)



#booleans
# type(False)
# print(1 > 2)
# print(1 == 1)
# b = None
# print(b)


#files
x = open('test.txt', 'w')
x.write('Hi')
x.close()