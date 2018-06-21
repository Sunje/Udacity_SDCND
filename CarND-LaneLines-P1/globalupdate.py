import globalvariables

def update_line(lines):
    del globalvariables.previous_lines
    globalvariables.previous_lines = lines


def update_left(x,y):
    del globalvariables.previous_left_x[:]
    del globalvariables.previous_left_y[:]
    globalvariables.previous_left_x.append(x)
    globalvariables.previous_left_y.append(y)

    
def update_right(x,y):
    del globalvariables.previous_right_x[:]
    del globalvariables.previous_right_y[:]
    globalvariables.previous_right_x.append(x)
    globalvariables.previous_right_y.append(y)



