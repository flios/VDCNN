def test_fun(a,b,**kwagrs):
    if kwagrs is not None:
        # a = kwagrs.pop('a',None)
        # b = kwagrs.pop('b',None)
        c = kwagrs.pop('c',5)
        d = kwagrs.pop('d',None)

    print(a,b,c,d);

test_dict = {
        'a':1,
        'b':2,
        'c':3,
        'e':50

        }

test_fun(1,2)
test_fun(**test_dict)
# test_fun(2,3,**test_dict)
