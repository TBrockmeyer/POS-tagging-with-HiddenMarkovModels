def txtfile_to_list (path)
    txtfile = open(path, 'r')
    resultlist = [line.split(',') for line in txtfile.readlines()]
    return resultlist
    
sentences_list = txtfile_to_list()